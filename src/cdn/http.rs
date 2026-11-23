//! HTTP/1.1 over TLS, as much of it as fetching from Nintendo's content
//! delivery network takes: `GET` and `HEAD` on one kept-alive connection per
//! host, bodies by length, chunked or up to the end of the connection.

use std::{
    io::{self, BufRead, BufReader, Read, Write},
    net::TcpStream,
    sync::Arc,
    time::Duration,
};

use rustls::{ClientConfig, ClientConnection, StreamOwned, pki_types::ServerName};

const PORT: u16 = 443;
/// How long a connection, a read or a write may stall.
const TIMEOUT: Duration = Duration::from_secs(60);
/// Longest status, header or chunk-size line taken.
const LINE_LIMIT: u64 = 16 * 1024;

type Stream = BufReader<StreamOwned<ClientConnection, TcpStream>>;

/// How the body of a response is delimited.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Body {
    Empty,
    Length(u64),
    Chunked,
    /// The server closes the connection after it.
    UntilClose,
}

/// The head of a response; [`Connection::read_body`] reads the rest.
#[derive(Debug)]
pub struct Response {
    pub status: u16,
    /// Names in lower case.
    headers: Vec<(String, String)>,
    body: Body,
    /// The connection cannot carry another request after this one.
    close: bool,
}

impl Response {
    /// The first header named `name`, in lower case.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(known, _)| known == name)
            .map(|(_, value)| value.as_str())
    }
}

/// A connection to one host, opened when the first request needs it and
/// again when the server has closed it.
pub struct Connection {
    host: &'static str,
    config: Arc<ClientConfig>,
    stream: Option<Stream>,
}

impl Connection {
    pub fn new(config: Arc<ClientConfig>, host: &'static str) -> Self {
        Self {
            host,
            config,
            stream: None,
        }
    }

    pub fn host(&self) -> &'static str {
        self.host
    }

    /// Sends `method path` with `headers` and reads the head of the answer.
    /// A request on a kept-alive connection that fails is sent once more on
    /// a new one: the server may have closed it while it was idle.
    pub fn send(
        &mut self,
        method: &str,
        path: &str,
        headers: &[(&str, &str)],
    ) -> io::Result<Response> {
        let mut request = format!("{method} {path} HTTP/1.1\r\nHost: {}\r\n", self.host);
        for (name, value) in headers {
            request += &format!("{name}: {value}\r\n");
        }
        request += "\r\n";
        let head = method == "HEAD";
        let reused = self.stream.is_some();
        match self.exchange(request.as_bytes(), head) {
            Err(_) if reused => self.exchange(request.as_bytes(), head),
            result => result,
        }
    }

    fn exchange(&mut self, request: &[u8], head: bool) -> io::Result<Response> {
        let result = self.open().and_then(|stream| {
            stream.get_mut().write_all(request)?;
            stream.get_mut().flush()?;
            read_head(stream, head)
        });
        if result.is_err() {
            self.stream = None;
        }
        result
    }

    fn open(&mut self) -> io::Result<&mut Stream> {
        if self.stream.is_none() {
            let name = ServerName::try_from(self.host).map_err(io::Error::other)?;
            let tls = ClientConnection::new(self.config.clone(), name).map_err(io::Error::other)?;
            let tcp = TcpStream::connect((self.host, PORT))?;
            tcp.set_read_timeout(Some(TIMEOUT))?;
            tcp.set_write_timeout(Some(TIMEOUT))?;
            tcp.set_nodelay(true)?;
            self.stream = Some(BufReader::new(StreamOwned::new(tls, tcp)));
        }
        Ok(self.stream.as_mut().expect("just opened"))
    }

    /// Copies the body of `response`, the last [`Self::send`] answered,
    /// into `sink`; how many bytes it held.
    pub fn read_body(&mut self, response: &Response, sink: &mut impl Write) -> io::Result<u64> {
        let stream = self
            .stream
            .as_mut()
            .expect("a response came on this connection");
        let result = read_body(stream, response.body, sink);
        if result.is_err() || response.close {
            self.stream = None;
        }
        result
    }

    /// Drops the connection, with whatever of a response is still unread.
    pub fn close(&mut self) {
        self.stream = None;
    }
}

/// One line without its CRLF.
fn read_line(reader: &mut impl BufRead, line: &mut String) -> io::Result<()> {
    line.clear();
    if reader.take(LINE_LIMIT).read_line(line)? == 0 {
        return Err(io::ErrorKind::UnexpectedEof.into());
    }
    if !line.ends_with('\n') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "an HTTP line is too long",
        ));
    }
    line.truncate(line.trim_end_matches(['\r', '\n']).len());
    Ok(())
}

fn invalid(what: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, what)
}

/// Status line and headers; `head` when the request was `HEAD`, whose
/// answer has no body whatever its headers say.
fn read_head(reader: &mut impl BufRead, head: bool) -> io::Result<Response> {
    let mut line = String::new();
    read_line(reader, &mut line)?;
    let status = line
        .strip_prefix("HTTP/1.")
        .and_then(|rest| rest.split(' ').nth(1))
        .and_then(|status| status.parse().ok())
        .ok_or_else(|| invalid("not an HTTP/1.x status line"))?;
    let mut headers = Vec::new();
    loop {
        read_line(reader, &mut line)?;
        if line.is_empty() {
            break;
        }
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| invalid("a header line without a colon"))?;
        headers.push((name.trim().to_ascii_lowercase(), value.trim().to_owned()));
    }
    let mut response = Response {
        status,
        headers,
        body: Body::Empty,
        close: false,
    };
    response.body = if head || (100..200).contains(&status) || status == 204 || status == 304 {
        Body::Empty
    } else if response
        .header("transfer-encoding")
        .is_some_and(|coding| coding.to_ascii_lowercase().contains("chunked"))
    {
        Body::Chunked
    } else if let Some(length) = response.header("content-length") {
        Body::Length(
            length
                .parse()
                .map_err(|_| invalid("a Content-Length that is not a number"))?,
        )
    } else {
        Body::UntilClose
    };
    response.close = response.body == Body::UntilClose
        || response
            .header("connection")
            .is_some_and(|connection| connection.eq_ignore_ascii_case("close"));
    Ok(response)
}

/// Exactly `length` bytes from `reader` into `sink`.
fn copy_exactly(reader: &mut impl BufRead, length: u64, sink: &mut impl Write) -> io::Result<()> {
    if io::copy(&mut reader.take(length), sink)? != length {
        return Err(io::ErrorKind::UnexpectedEof.into());
    }
    Ok(())
}

fn read_body(reader: &mut impl BufRead, body: Body, sink: &mut impl Write) -> io::Result<u64> {
    match body {
        Body::Empty => Ok(0),
        Body::Length(length) => copy_exactly(reader, length, sink).map(|()| length),
        Body::UntilClose => io::copy(reader, sink),
        Body::Chunked => {
            let mut line = String::new();
            let mut total = 0;
            loop {
                read_line(reader, &mut line)?;
                let digits = line.split(';').next().unwrap_or_default().trim();
                let size = u64::from_str_radix(digits, 16)
                    .map_err(|_| invalid("a chunk size that is not a hex number"))?;
                if size == 0 {
                    // Trailers, up to the empty line.
                    loop {
                        read_line(reader, &mut line)?;
                        if line.is_empty() {
                            return Ok(total);
                        }
                    }
                }
                copy_exactly(reader, size, sink)?;
                total += size;
                read_line(reader, &mut line)?;
                if !line.is_empty() {
                    return Err(invalid("a chunk longer than its size"));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response(text: &[u8], head: bool) -> (Response, Vec<u8>, usize) {
        let mut reader = io::Cursor::new(text);
        let response = read_head(&mut reader, head).unwrap();
        let mut body = Vec::new();
        read_body(&mut reader, response.body, &mut body).unwrap();
        (response, body, reader.position() as usize)
    }

    #[test]
    fn chunked_bodies_join_up_and_the_next_response_starts_after_the_trailers() {
        let text = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n\
            4\r\nWiki\r\n5;name=value\r\npedia\r\n0\r\nTrailer: x\r\n\r\nHTTP/1.1";
        let (response, body, end) = response(text, false);
        assert_eq!(body, b"Wikipedia");
        assert_eq!(&text[end..], b"HTTP/1.1");
        assert!(!response.close);
    }

    #[test]
    fn a_head_answer_has_no_body_whatever_its_length_says() {
        let text = b"HTTP/1.1 200 OK\r\nX-Nintendo-Content-ID: 0123\r\n\
            Content-Length: 3584\r\nConnection: close\r\n\r\nHTTP/1.1";
        let (response, body, end) = response(text, true);
        assert!(body.is_empty());
        assert_eq!(&text[end..], b"HTTP/1.1");
        assert_eq!(response.header("x-nintendo-content-id"), Some("0123"));
        assert!(response.close);
    }
}

//! A gamepad of the host at the console's controls: the user's own Joy-Con,
//! a Pro Controller or another pad paired to the Mac, read through Apple's
//! GameController framework and turned into what the emulated pair of
//! Joy-Con is made to do ([`Pad`], [`crate::machine::Machine::set_pad`]).
//! The emulated Joy-Con stay where they are (on the rails or off); the
//! host's pad holds their buttons and pushes their sticks, as the keys of
//! the emulator's window do.
//!
//! # What the framework gives
//!
//! `GCController.controllers` lists the pads connected; each has a
//! `physicalInputProfile` whose `buttons` and `dpads` dictionaries name
//! its elements (`GCInputButtonA` is "Button A", and so on: Apple,
//! *GameController/GCInputNames.h*), read here by polling
//! (`GCControllerButtonInput.isPressed`, `GCControllerAxisInput.value`).
//!
//! Observed on macOS 26.6 with a pair of Joy-Con paired over Bluetooth:
//!
//! * The list is empty in a process whose main thread runs no run loop,
//!   and has the pad once the main run loop has run, in a command-line
//!   process that is never the active application too. So [`HostPads`]
//!   belongs to the main thread of a program that runs one (a window's
//!   event loop does).
//! * Two Joy-Con are one controller, `productCategory` "Nintendo Switch
//!   Joy-Con (L/R)", `vendorName` "Joy-Con (L/R)", with the elements
//!   Button A, B, X, Y, Home, Menu, Options, Left and Right Shoulder, Left
//!   and Right Trigger, Left and Right Thumbstick Button, Direction Pad,
//!   Left and Right Thumbstick. There is no "Button Share": the Capture
//!   button of a fused pair is not to be had.
//!
//! The other categories are the framework's own strings ("Nintendo Switch
//! Joy-Con (L)", "Nintendo Switch Joy-Con (R)", "Switch Pro Controller"
//! are in the GameController binary, and SDL's
//! *src/joystick/apple/SDL_mfijoystick.m* tells the pads apart by them).
//!
//! # Which element is which button
//!
//! * A pair of Joy-Con and a Pro Controller ([`Kind::Nintendo`]): the
//!   framework names the face buttons as Nintendo labels them, Button A
//!   the right one (SDL maps `GCInputButtonA` to its east button for
//!   these, `has_nintendo_buttons`), so A, B, X, Y are themselves. Left
//!   and Right Shoulder are L and R, the triggers ZL and ZR, Menu is +,
//!   Options is -, Share (where there is one) Capture.
//! * Any other pad ([`Kind::Other`]: Xbox, PlayStation, MFi): Button A is
//!   the bottom face button, B the right, X the left, Y the top (Apple,
//!   *GCExtendedGamepad*), and the console's buttons are given by place:
//!   the right one is A, the bottom B, the top X, the left Y. So a game's
//!   "press A" is the right-hand button of every pad, not the one an
//!   Xbox pad labels A.
//! * A Joy-Con alone ([`Kind::JoyConLeft`], [`Kind::JoyConRight`]) is
//!   that half of the pair, held upright. \[inference\] None was
//!   connected alone here; what follows is from the joycon-keys project's
//!   *Sources/JoyConKeys/Engine/GCBackend.swift*, written against the
//!   hardware. A lone Joy-Con has no extended gamepad profile, only
//!   Button A, B, X, Y, Menu, Home (right) or Share (left), a Direction
//!   Pad that is its stick, and Left and Right Shoulder that are its rail
//!   buttons SL and SR. The framework reports it as held sideways, so the
//!   stick is turned back: upright up is +x of a right Joy-Con and -x of
//!   a left one, upright right is -y and +y. The left Joy-Con's arrow
//!   buttons come as the face button in the same place of a right one (X
//!   up, A right, B down, Y left). L, ZL, R, ZR and the stick clicks of a
//!   lone Joy-Con are not reported at all (SDL issue 6095), so its rail
//!   buttons stand in: the one nearer the top (SL on the left Joy-Con, SR
//!   on the right) is L or R, the other ZL or ZR. Pair both Joy-Con for
//!   the real ones.
//!
//! macOS takes some buttons for itself (Home opens its own overlay);
//! each element bound to such a system gesture is asked to be left to the
//! program (`preferredSystemGestureState`, as SDL does). SDL notes that a
//! fused pair's Home never shows as held all the same.
//!
//! Several pads at once are one pair of hands: a button any of them holds
//! is held, and a stick is where the pad that pushes it furthest has it.

use objc2::rc::{Retained, autoreleasepool};
use objc2_foundation::NSString;
use objc2_game_controller::{
    GCController, GCControllerAxisInput, GCControllerButtonInput, GCDevice, GCSystemGestureState,
};

use crate::soc::joycon::{Button, Pad, Stick};

/// What a host controller is, which decides what its elements mean (the
/// module's text has the sources).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// A left Joy-Con alone: the left half of the pair.
    JoyConLeft,
    /// A right Joy-Con alone: the right half.
    JoyConRight,
    /// Both Joy-Con as one controller, or a Pro Controller: face buttons
    /// named as labelled.
    Nintendo,
    /// Anything else: face buttons named by place, Button A the bottom.
    Other,
}

impl Kind {
    /// From a controller's `productCategory`.
    fn of(product_category: &str) -> Self {
        match product_category {
            "Nintendo Switch Joy-Con (L)" => Self::JoyConLeft,
            "Nintendo Switch Joy-Con (R)" => Self::JoyConRight,
            "Nintendo Switch Joy-Con (L/R)" | "Switch Pro Controller" => Self::Nintendo,
            _ => Self::Other,
        }
    }

    /// What it is to the console, for the line that says it connected.
    fn drives(self) -> &'static str {
        match self {
            Self::JoyConLeft => "the left Joy-Con",
            Self::JoyConRight => "the right Joy-Con",
            Self::Nintendo => "both Joy-Con",
            Self::Other => "both Joy-Con, A B X Y by place",
        }
    }

    /// The console's button that `input` of this kind of controller holds.
    fn button(self, input: Input) -> Option<Button> {
        use Input as I;
        Some(match (self, input) {
            (Self::Nintendo | Self::JoyConRight, I::A) => Button::A,
            (Self::Nintendo | Self::JoyConRight, I::B) => Button::B,
            (Self::Nintendo | Self::JoyConRight, I::X) => Button::X,
            (Self::Nintendo | Self::JoyConRight, I::Y) => Button::Y,
            (Self::Other, I::A) => Button::B,
            (Self::Other, I::B) => Button::A,
            (Self::Other, I::X) => Button::Y,
            (Self::Other, I::Y) => Button::X,
            (Self::JoyConLeft, I::X) => Button::UP,
            (Self::JoyConLeft, I::A) => Button::RIGHT,
            (Self::JoyConLeft, I::B) => Button::DOWN,
            (Self::JoyConLeft, I::Y) => Button::LEFT,
            // The rail buttons of a Joy-Con alone: SL and SR.
            (Self::JoyConLeft, I::LeftShoulder) => Button::L,
            (Self::JoyConLeft, I::RightShoulder) => Button::ZL,
            (Self::JoyConRight, I::RightShoulder) => Button::R,
            (Self::JoyConRight, I::LeftShoulder) => Button::ZR,
            // Its one of + and -, whichever name it comes under.
            (Self::JoyConLeft, I::Menu | I::Options) => Button::MINUS,
            (Self::JoyConRight, I::Menu | I::Options) => Button::PLUS,
            (Self::JoyConLeft, I::Share) => Button::CAPTURE,
            (Self::JoyConRight, I::Home) => Button::HOME,
            (Self::JoyConLeft, I::LeftStickButton) => Button::LEFT_STICK,
            (Self::JoyConRight, I::RightStickButton) => Button::RIGHT_STICK,
            (Self::JoyConLeft | Self::JoyConRight, _) => return None,
            (_, I::LeftShoulder) => Button::L,
            (_, I::RightShoulder) => Button::R,
            (_, I::LeftTrigger) => Button::ZL,
            (_, I::RightTrigger) => Button::ZR,
            (_, I::Menu) => Button::PLUS,
            (_, I::Options) => Button::MINUS,
            (_, I::Home) => Button::HOME,
            (_, I::Share) => Button::CAPTURE,
            (_, I::LeftStickButton) => Button::LEFT_STICK,
            (_, I::RightStickButton) => Button::RIGHT_STICK,
            (_, I::Up) => Button::UP,
            (_, I::Down) => Button::DOWN,
            (_, I::Left) => Button::LEFT,
            (_, I::Right) => Button::RIGHT,
        })
    }
}

/// A button of a controller's physical input profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Input {
    A,
    B,
    X,
    Y,
    LeftShoulder,
    RightShoulder,
    LeftTrigger,
    RightTrigger,
    Menu,
    Options,
    Home,
    Share,
    LeftStickButton,
    RightStickButton,
    Up,
    Down,
    Left,
    Right,
}

impl Input {
    const ALL: [Self; 18] = [
        Self::A,
        Self::B,
        Self::X,
        Self::Y,
        Self::LeftShoulder,
        Self::RightShoulder,
        Self::LeftTrigger,
        Self::RightTrigger,
        Self::Menu,
        Self::Options,
        Self::Home,
        Self::Share,
        Self::LeftStickButton,
        Self::RightStickButton,
        Self::Up,
        Self::Down,
        Self::Left,
        Self::Right,
    ];

    /// Its key in `GCPhysicalInputProfile.buttons`: the values of the
    /// `GCInput...` constants, and the four buttons of the direction pad
    /// as the profile lists them.
    fn name(self) -> &'static str {
        match self {
            Self::A => "Button A",
            Self::B => "Button B",
            Self::X => "Button X",
            Self::Y => "Button Y",
            Self::LeftShoulder => "Left Shoulder",
            Self::RightShoulder => "Right Shoulder",
            Self::LeftTrigger => "Left Trigger",
            Self::RightTrigger => "Right Trigger",
            Self::Menu => "Button Menu",
            Self::Options => "Button Options",
            Self::Home => "Button Home",
            Self::Share => "Button Share",
            Self::LeftStickButton => "Left Thumbstick Button",
            Self::RightStickButton => "Right Thumbstick Button",
            Self::Up => "Direction Pad Up",
            Self::Down => "Direction Pad Down",
            Self::Left => "Direction Pad Left",
            Self::Right => "Direction Pad Right",
        }
    }
}

/// The two-axis elements read, by their keys in
/// `GCPhysicalInputProfile.dpads`; a [`Reading`] has them in this order.
const AXES: [&str; 3] = ["Direction Pad", "Left Thumbstick", "Right Thumbstick"];
const DIRECTION_PAD: usize = 0;
const LEFT_THUMBSTICK: usize = 1;
const RIGHT_THUMBSTICK: usize = 2;

/// What a controller's elements read at one moment.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct Reading {
    /// The buttons pressed, a bit per [`Input`] (its place in the enum).
    pressed: u32,
    /// x and y of each of [`AXES`], from -1 to 1, right and up positive.
    axes: [(f32, f32); 3],
}

impl Reading {
    fn pressed(&self, input: Input) -> bool {
        self.pressed & 1 << input as u32 != 0
    }

    /// What a controller of `kind` that reads this does to the Joy-Con.
    fn pad(&self, kind: Kind) -> Pad {
        let buttons = Input::ALL
            .into_iter()
            .filter(|&input| self.pressed(input))
            .filter_map(|input| kind.button(input))
            .fold(0, |held, button| held | button.bits());
        // A Joy-Con alone reports its stick as the direction pad of a pad
        // held sideways, the rail up: turned back upright here.
        let (x, y) = self.axes[DIRECTION_PAD];
        let mut sticks = [(0.0, 0.0); 2];
        match kind {
            Kind::JoyConLeft => sticks[Stick::Left as usize] = (y, -x),
            Kind::JoyConRight => sticks[Stick::Right as usize] = (-y, x),
            Kind::Nintendo | Kind::Other => {
                sticks = [self.axes[LEFT_THUMBSTICK], self.axes[RIGHT_THUMBSTICK]];
            }
        }
        Pad { buttons, sticks }
    }

    /// The elements away from rest by their names, for the trace.
    fn described(&self) -> String {
        let mut parts: Vec<String> = Input::ALL
            .into_iter()
            .filter(|&input| self.pressed(input))
            .map(|input| input.name().to_string())
            .collect();
        for (name, (x, y)) in AXES.into_iter().zip(self.axes) {
            if (x, y) != (0.0, 0.0) {
                parts.push(format!("{name} ({x:.2}, {y:.2})"));
            }
        }
        match parts.is_empty() {
            true => "nothing".into(),
            false => parts.join(", "),
        }
    }
}

/// `pad`, for the trace, in the words of scripts: the buttons by name and
/// where each stick is.
fn described(pad: &Pad) -> String {
    let mut parts: Vec<String> = Button::names(pad.buttons).map(String::from).collect();
    for (name, (x, y)) in ["left", "right"].into_iter().zip(pad.sticks) {
        if (x, y) != (0.0, 0.0) {
            parts.push(format!("stick {name} {x:.2} {y:.2}"));
        }
    }
    match parts.is_empty() {
        true => "nothing".into(),
        false => parts.join(", "),
    }
}

/// What two pads do together: the buttons of both, and each stick where
/// the one that pushes it further has it.
fn together(one: Pad, other: Pad) -> Pad {
    let further = |a: (f32, f32), b: (f32, f32)| {
        match a.0 * a.0 + a.1 * a.1 >= b.0 * b.0 + b.1 * b.1 {
            true => a,
            false => b,
        }
    };
    Pad {
        buttons: one.buttons | other.buttons,
        sticks: [
            further(one.sticks[0], other.sticks[0]),
            further(one.sticks[1], other.sticks[1]),
        ],
    }
}

/// A connected controller and those of its elements that mean something.
struct Bound {
    controller: Retained<GCController>,
    name: String,
    kind: Kind,
    buttons: Vec<(Input, Retained<GCControllerButtonInput>)>,
    /// The x and y axis of each of [`AXES`] it has.
    axes: [Option<[Retained<GCControllerAxisInput>; 2]>; 3],
    /// What it last read, for the trace.
    last: Reading,
}

impl Bound {
    fn new(controller: Retained<GCController>) -> Self {
        // SAFETY: plain property reads of a live controller and its
        // profile, on the thread the framework's objects are used from.
        let (name, category, profile) = unsafe {
            (
                controller.vendorName(),
                controller.productCategory().to_string(),
                controller.physicalInputProfile(),
            )
        };
        let (named_buttons, named_axes) = unsafe { (profile.buttons(), profile.dpads()) };
        let buttons = Input::ALL
            .into_iter()
            .filter_map(|input| {
                let button = named_buttons.objectForKey(&NSString::from_str(input.name()))?;
                // What macOS does with the button itself (HOME's overlay)
                // would keep it from the console.
                unsafe {
                    if button.isBoundToSystemGesture() {
                        button.setPreferredSystemGestureState(GCSystemGestureState::Disabled);
                    }
                }
                Some((input, button))
            })
            .collect();
        let axes = AXES.map(|name| {
            let pad = named_axes.objectForKey(&NSString::from_str(name))?;
            Some(unsafe { [pad.xAxis(), pad.yAxis()] })
        });
        Self {
            controller,
            name: name.map_or_else(|| category.clone(), |name| name.to_string()),
            kind: Kind::of(&category),
            buttons,
            axes,
            last: Reading::default(),
        }
    }

    fn read(&self) -> Reading {
        // SAFETY: as in `new`; the elements belong to `controller`, kept.
        let pressed = self
            .buttons
            .iter()
            .filter(|(_, button)| unsafe { button.isPressed() })
            .fold(0, |pressed, &(input, _)| pressed | 1 << input as u32);
        let axes = [0, 1, 2].map(|index| match &self.axes[index] {
            Some([x, y]) => unsafe { (x.value(), y.value()) },
            None => (0.0, 0.0),
        });
        Reading { pressed, axes }
    }

    /// The names of the elements read, for the trace.
    fn elements(&self) -> String {
        let buttons = self.buttons.iter().map(|(input, _)| input.name());
        let axes = AXES
            .into_iter()
            .zip(&self.axes)
            .filter_map(|(name, axes)| axes.as_ref().map(|_| name));
        buttons.chain(axes).collect::<Vec<_>>().join(", ")
    }
}

/// The host's gamepads, as [`HostPads::poll`] last found them. It belongs
/// to the main thread, whose run loop the framework finds controllers on
/// (the module's text): the thread of the emulator's window, or the UI's.
pub struct HostPads {
    bound: Vec<Bound>,
    /// Whether each change of what a controller reads is told.
    trace: bool,
}

impl HostPads {
    /// With `trace`, [`HostPads::poll`] tells what each controller reads
    /// and what that does, whenever it changes.
    pub fn new(trace: bool) -> Self {
        Self {
            bound: Vec::new(),
            trace,
        }
    }

    /// What the connected pads do to the Joy-Con now; nothing without
    /// one, and nothing unless `plays`: whether they are the console's to
    /// be played with at the moment, which is for the caller's window to
    /// say (the one in front is played, as with its keys), the framework
    /// being told to pass input on wherever the program is. `tell` is
    /// given a line for each pad that came or went since the last call (a
    /// pad that went has let go of everything).
    pub fn poll(&mut self, plays: bool, mut tell: impl FnMut(String)) -> Pad {
        autoreleasepool(|_| {
            // SAFETY: a class property, read on the main thread.
            let controllers = unsafe { GCController::controllers() };
            let same = |bound: &Bound, controller: &GCController| {
                std::ptr::eq(Retained::as_ptr(&bound.controller), controller)
            };
            if controllers.len() != self.bound.len()
                || !controllers
                    .iter()
                    .zip(&self.bound)
                    .all(|(controller, bound)| same(bound, &controller))
            {
                self.rebind(controllers.iter(), &mut tell);
            }
            let mut pad = Pad::default();
            for bound in &mut self.bound {
                let reading = bound.read();
                let does = match plays {
                    true => reading.pad(bound.kind),
                    false => Pad::default(),
                };
                if self.trace && reading != bound.last {
                    let does = match plays {
                        true => described(&does),
                        false => "left alone".into(),
                    };
                    tell(format!(
                        "host gamepad: {}: {} => {does}",
                        bound.name,
                        reading.described()
                    ));
                }
                bound.last = reading;
                pad = together(pad, does);
            }
            pad
        })
    }

    /// Takes `controllers` as the pads connected, keeping those known.
    fn rebind(
        &mut self,
        controllers: impl Iterator<Item = Retained<GCController>>,
        tell: &mut impl FnMut(String),
    ) {
        let mut known = std::mem::take(&mut self.bound);
        if known.is_empty() {
            // Input reaches the program while another application is in
            // front as well, so that whether a pad drives the console is
            // for the caller's window to say, as with its keys. Set once a
            // controller is there: SDL found macOS 12.3 to crash on this
            // before the first one connected (its commit d1a3981).
            if objc2::available!(macos = 11.3) {
                unsafe { GCController::setShouldMonitorBackgroundEvents(true) };
            }
        }
        for controller in controllers {
            let place = known
                .iter()
                .position(|bound| std::ptr::eq(Retained::as_ptr(&bound.controller), &*controller));
            let bound = match place {
                Some(place) => known.swap_remove(place),
                None => {
                    let bound = Bound::new(controller);
                    tell(format!(
                        "host gamepad: {} connected, drives {}",
                        bound.name,
                        bound.kind.drives()
                    ));
                    if self.trace {
                        tell(format!(
                            "host gamepad: {}: reads {}",
                            bound.name,
                            bound.elements()
                        ));
                    }
                    bound
                }
            };
            self.bound.push(bound);
        }
        for gone in known {
            tell(format!("host gamepad: {} disconnected", gone.name));
        }
    }

    /// The names of the pads connected, as last polled.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.bound.iter().map(|bound| bound.name.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reading(pressed: &[Input], axes: [(f32, f32); 3]) -> Reading {
        Reading {
            pressed: pressed
                .iter()
                .fold(0, |pressed, &input| pressed | 1 << input as u32),
            axes,
        }
    }

    fn held(buttons: &[Button]) -> u32 {
        buttons.iter().fold(0, |held, button| held | button.bits())
    }

    const REST: [(f32, f32); 3] = [(0.0, 0.0); 3];

    #[test]
    fn controllers_are_told_apart_by_product_category() {
        assert_eq!(Kind::of("Nintendo Switch Joy-Con (L)"), Kind::JoyConLeft);
        assert_eq!(Kind::of("Nintendo Switch Joy-Con (R)"), Kind::JoyConRight);
        assert_eq!(Kind::of("Nintendo Switch Joy-Con (L/R)"), Kind::Nintendo);
        assert_eq!(Kind::of("Switch Pro Controller"), Kind::Nintendo);
        for other in ["Xbox One", "DualSense", "DualShock 4", "MFi"] {
            assert_eq!(Kind::of(other), Kind::Other);
        }
    }

    #[test]
    fn a_pair_of_joycon_is_the_whole_pad_as_labelled() {
        use Input as I;
        let all = reading(&Input::ALL, [(0.0, 0.0), (0.25, -1.0), (-0.5, 0.75)]);
        let pad = all.pad(Kind::Nintendo);
        // Every button there is.
        assert_eq!(Button::names(pad.buttons).count(), 18);
        assert_eq!(pad.sticks, [(0.25, -1.0), (-0.5, 0.75)]);
        for (input, button) in [
            (I::A, Button::A),
            (I::B, Button::B),
            (I::X, Button::X),
            (I::Y, Button::Y),
            (I::LeftShoulder, Button::L),
            (I::RightShoulder, Button::R),
            (I::LeftTrigger, Button::ZL),
            (I::RightTrigger, Button::ZR),
            (I::Menu, Button::PLUS),
            (I::Options, Button::MINUS),
            (I::Home, Button::HOME),
            (I::Share, Button::CAPTURE),
            (I::LeftStickButton, Button::LEFT_STICK),
            (I::RightStickButton, Button::RIGHT_STICK),
            (I::Up, Button::UP),
            (I::Down, Button::DOWN),
            (I::Left, Button::LEFT),
            (I::Right, Button::RIGHT),
        ] {
            let pad = reading(&[input], REST).pad(Kind::Nintendo);
            assert_eq!(pad.buttons, button.bits(), "{input:?}");
        }
    }

    #[test]
    fn another_pad_gives_the_face_buttons_by_place() {
        use Input as I;
        for (input, button) in [
            // Button A is the bottom one, where a Joy-Con has B.
            (I::A, Button::B),
            (I::B, Button::A),
            (I::X, Button::Y),
            (I::Y, Button::X),
            (I::LeftTrigger, Button::ZL),
            (I::Menu, Button::PLUS),
            (I::Up, Button::UP),
        ] {
            let pad = reading(&[input], REST).pad(Kind::Other);
            assert_eq!(pad.buttons, button.bits(), "{input:?}");
        }
        let pushed = reading(&[], [(0.0, 0.0), (1.0, 0.0), (0.0, -1.0)]);
        assert_eq!(pushed.pad(Kind::Other).sticks, [(1.0, 0.0), (0.0, -1.0)]);
    }

    #[test]
    fn a_left_joycon_alone_is_the_left_half_upright() {
        use Input as I;
        for (input, button) in [
            (I::X, Button::UP),
            (I::A, Button::RIGHT),
            (I::B, Button::DOWN),
            (I::Y, Button::LEFT),
            (I::LeftShoulder, Button::L),
            (I::RightShoulder, Button::ZL),
            (I::Menu, Button::MINUS),
            (I::Options, Button::MINUS),
            (I::Share, Button::CAPTURE),
            (I::LeftStickButton, Button::LEFT_STICK),
        ] {
            let pad = reading(&[input], REST).pad(Kind::JoyConLeft);
            assert_eq!(pad.buttons, button.bits(), "{input:?}");
        }
        // The stick's own directions held as buttons are not the arrows.
        let pad = reading(&[I::Up, I::Home, I::RightTrigger], REST).pad(Kind::JoyConLeft);
        assert_eq!(pad.buttons, 0);
        // Upright up is reported as -x, upright right as +y.
        let up = reading(&[], [(-1.0, 0.0), (0.0, 0.0), (0.0, 0.0)]);
        assert_eq!(up.pad(Kind::JoyConLeft).sticks, [(0.0, 1.0), (0.0, 0.0)]);
        let right = reading(&[], [(0.0, 1.0), (0.0, 0.0), (0.0, 0.0)]);
        assert_eq!(right.pad(Kind::JoyConLeft).sticks, [(1.0, 0.0), (0.0, 0.0)]);
    }

    #[test]
    fn a_right_joycon_alone_is_the_right_half_upright() {
        use Input as I;
        for (input, button) in [
            (I::A, Button::A),
            (I::B, Button::B),
            (I::X, Button::X),
            (I::Y, Button::Y),
            (I::RightShoulder, Button::R),
            (I::LeftShoulder, Button::ZR),
            (I::Menu, Button::PLUS),
            (I::Options, Button::PLUS),
            (I::Home, Button::HOME),
            (I::RightStickButton, Button::RIGHT_STICK),
        ] {
            let pad = reading(&[input], REST).pad(Kind::JoyConRight);
            assert_eq!(pad.buttons, button.bits(), "{input:?}");
        }
        // Upright up is reported as +x, upright right as -y.
        let up = reading(&[], [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0)]);
        assert_eq!(up.pad(Kind::JoyConRight).sticks, [(0.0, 0.0), (0.0, 1.0)]);
        let right = reading(&[], [(0.0, -1.0), (0.0, 0.0), (0.0, 0.0)]);
        assert_eq!(right.pad(Kind::JoyConRight).sticks, [(0.0, 0.0), (1.0, 0.0)]);
    }

    #[test]
    fn two_joycon_apart_are_one_pair_of_hands() {
        let left = reading(&[Input::X], [(-0.5, 0.0), (0.0, 0.0), (0.0, 0.0)]);
        let right = reading(&[Input::X], [(0.0, -1.0), (0.0, 0.0), (0.0, 0.0)]);
        let pad = together(left.pad(Kind::JoyConLeft), right.pad(Kind::JoyConRight));
        assert_eq!(pad.buttons, held(&[Button::UP, Button::X]));
        assert_eq!(pad.sticks, [(0.0, 0.5), (1.0, 0.0)]);
        // The stick pushed further is where the stick is.
        let nudged = Pad {
            buttons: 0,
            sticks: [(0.1, 0.0), (0.0, 0.0)],
        };
        assert_eq!(together(nudged, pad).sticks, pad.sticks);
        assert_eq!(together(pad, Pad::default()), pad);
    }

    #[test]
    fn what_a_pad_does_is_described_in_the_words_of_scripts() {
        let reading = reading(
            &[Input::A, Input::LeftTrigger],
            [(0.0, 0.0), (0.0, 1.0), (0.0, 0.0)],
        );
        assert_eq!(
            reading.described(),
            "Button A, Left Trigger, Left Thumbstick (0.00, 1.00)"
        );
        assert_eq!(
            described(&reading.pad(Kind::Nintendo)),
            "a, zl, stick left 0.00 1.00"
        );
        assert_eq!(described(&Pad::default()), "nothing");
    }

    /// The framework's own controller object, of the kind a program makes
    /// to replay input (`controllerWithExtendedGamepad`), is bound and
    /// read as a connected one is: its elements are found under the names
    /// used here.
    #[test]
    fn a_controller_of_the_framework_is_bound_and_read() {
        // SAFETY: a snapshot controller of this test's own, whose
        // elements take the values set.
        let controller = unsafe { GCController::controllerWithExtendedGamepad() };
        let bound = Bound::new(controller);
        assert_eq!(bound.kind, Kind::Other);
        let has = |input: Input| bound.buttons.iter().any(|&(bound, _)| bound == input);
        for input in [
            Input::A,
            Input::B,
            Input::X,
            Input::Y,
            Input::LeftShoulder,
            Input::RightShoulder,
            Input::LeftTrigger,
            Input::RightTrigger,
            Input::Up,
            Input::Down,
            Input::Left,
            Input::Right,
        ] {
            assert!(has(input), "{input:?} of {}", bound.elements());
        }
        assert!(bound.axes.iter().all(Option::is_some), "{}", bound.elements());
        assert_eq!(bound.read(), Reading::default());
        let profile = unsafe { bound.controller.physicalInputProfile() };
        let (buttons, sticks) = unsafe { (profile.buttons(), profile.dpads()) };
        let (a, stick) = (
            buttons.objectForKey(&NSString::from_str("Button A")).unwrap(),
            sticks.objectForKey(&NSString::from_str("Left Thumbstick")).unwrap(),
        );
        unsafe {
            a.setValue(1.0);
            stick.setValueForXAxis_yAxis(-1.0, 0.5);
        }
        let read = bound.read();
        assert_eq!(read, reading(&[Input::A], [(0.0, 0.0), (-1.0, 0.5), (0.0, 0.0)]));
        assert_eq!(
            read.pad(bound.kind),
            Pad {
                buttons: Button::B.bits(),
                sticks: [(-1.0, 0.5), (0.0, 0.0)],
            }
        );
    }
}

//! Color blending: the blend equations and factors of the 3D class's
//! `SET_BLEND_*` / `SET_BLEND_PER_TARGET_*` methods. Both the OpenGL-style
//! and the Direct3D-style encodings of factors and operations are accepted,
//! with the values of [public: NVIDIA open-gpu-doc `classes/3d/clb197.h`,
//! `NVB197_SET_BLEND_COLOR_SOURCE_COEFF_V_*`, `NVB197_SET_BLEND_COLOR_OP_V_*`];
//! the semantics are the OpenGL/Direct3D blend equations those names
//! denote. The netlist's initial state uses the Direct3D encoding (op 1
//! ADD, source 2 ONE, destination 1 ZERO; nvservices netlist region 7).

use super::Unmodeled;

/// A blend factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Factor {
    Zero,
    One,
    SourceColor,
    OneMinusSourceColor,
    SourceAlpha,
    OneMinusSourceAlpha,
    DestinationAlpha,
    OneMinusDestinationAlpha,
    DestinationColor,
    OneMinusDestinationColor,
    SourceAlphaSaturate,
    ConstantColor,
    OneMinusConstantColor,
    ConstantAlpha,
    OneMinusConstantAlpha,
    Source1Color,
    OneMinusSource1Color,
    Source1Alpha,
    OneMinusSource1Alpha,
}

impl Factor {
    /// Decodes a `*_COEFF` register value.
    pub fn from_raw(raw: u32) -> Result<Self, Unmodeled> {
        use Factor::*;
        Ok(match raw {
            0x4000 | 0x01 => Zero,
            0x4001 | 0x02 => One,
            0x4300 | 0x03 => SourceColor,
            0x4301 | 0x04 => OneMinusSourceColor,
            0x4302 | 0x05 => SourceAlpha,
            0x4303 | 0x06 => OneMinusSourceAlpha,
            0x4304 | 0x07 => DestinationAlpha,
            0x4305 | 0x08 => OneMinusDestinationAlpha,
            0x4306 | 0x09 => DestinationColor,
            0x4307 | 0x0A => OneMinusDestinationColor,
            0x4308 | 0x0B => SourceAlphaSaturate,
            0xC001 | 0x0E => ConstantColor,
            0xC002 | 0x0F => OneMinusConstantColor,
            0xC003 => ConstantAlpha,
            0xC004 => OneMinusConstantAlpha,
            0xC900 | 0x10 => Source1Color,
            0xC901 | 0x11 => OneMinusSource1Color,
            0xC902 | 0x12 => Source1Alpha,
            0xC903 | 0x13 => OneMinusSource1Alpha,
            // D3D BOTHSRCALPHA / BOTHINVSRCALPHA (0x0C, 0x0D) replace both
            // factors at once; not modeled.
            _ => {
                return Err(Unmodeled(format!("3D: blend factor {raw:#x}")));
            }
        })
    }

    /// Whether the factor reads the second color output of the fragment
    /// shader (dual-source blending).
    pub fn dual_source(self) -> bool {
        matches!(
            self,
            Factor::Source1Color
                | Factor::OneMinusSource1Color
                | Factor::Source1Alpha
                | Factor::OneMinusSource1Alpha
        )
    }
}

/// A blend operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

impl Operation {
    /// Decodes a `*_OP` register value.
    pub fn from_raw(raw: u32) -> Result<Self, Unmodeled> {
        Ok(match raw {
            0x8006 | 1 => Operation::Add,
            0x800A | 2 => Operation::Subtract,
            0x800B | 3 => Operation::ReverseSubtract,
            0x8007 | 4 => Operation::Min,
            0x8008 | 5 => Operation::Max,
            _ => return Err(Unmodeled(format!("3D: blend operation {raw:#x}"))),
        })
    }
}

/// The blend state of one render target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Blend {
    pub color_op: Operation,
    pub color_source: Factor,
    pub color_destination: Factor,
    pub alpha_op: Operation,
    pub alpha_source: Factor,
    pub alpha_destination: Factor,
}

/// The inputs of one blend.
#[derive(Debug, Clone, Copy)]
pub struct Inputs {
    /// Fragment shader output for this target.
    pub source: [f32; 4],
    /// Second fragment output (dual-source blending).
    pub source1: [f32; 4],
    /// Current render target value.
    pub destination: [f32; 4],
    pub constant: [f32; 4],
    /// `SET_BLEND_FLOAT_OPTION` `ZERO_TIMES_ANYTHING_IS_ZERO`: a zero
    /// factor or color yields 0 even against infinity or NaN (NVN's init
    /// blast sets it).
    pub zero_times_anything_is_zero: bool,
}

fn factor(factor: Factor, inputs: &Inputs, component: usize) -> f32 {
    let source = inputs.source;
    let destination = inputs.destination;
    let constant = inputs.constant;
    let source1 = inputs.source1;
    match factor {
        Factor::Zero => 0.0,
        Factor::One => 1.0,
        Factor::SourceColor => source[component],
        Factor::OneMinusSourceColor => 1.0 - source[component],
        Factor::SourceAlpha => source[3],
        Factor::OneMinusSourceAlpha => 1.0 - source[3],
        Factor::DestinationAlpha => destination[3],
        Factor::OneMinusDestinationAlpha => 1.0 - destination[3],
        Factor::DestinationColor => destination[component],
        Factor::OneMinusDestinationColor => 1.0 - destination[component],
        // f = min(As, 1 - Ad) for RGB, 1 for alpha (OpenGL/Direct3D).
        Factor::SourceAlphaSaturate => {
            if component == 3 {
                1.0
            } else {
                source[3].min(1.0 - destination[3])
            }
        }
        Factor::ConstantColor => constant[component],
        Factor::OneMinusConstantColor => 1.0 - constant[component],
        Factor::ConstantAlpha => constant[3],
        Factor::OneMinusConstantAlpha => 1.0 - constant[3],
        Factor::Source1Color => source1[component],
        Factor::OneMinusSource1Color => 1.0 - source1[component],
        Factor::Source1Alpha => source1[3],
        Factor::OneMinusSource1Alpha => 1.0 - source1[3],
    }
}

fn combine(op: Operation, s: f32, sf: f32, d: f32, df: f32, zero_rule: bool) -> f32 {
    let mul = |a: f32, b: f32| {
        if zero_rule && (a == 0.0 || b == 0.0) {
            0.0
        } else {
            a * b
        }
    };
    match op {
        Operation::Add => mul(s, sf) + mul(d, df),
        Operation::Subtract => mul(s, sf) - mul(d, df),
        Operation::ReverseSubtract => mul(d, df) - mul(s, sf),
        // Min and max ignore the factors (OpenGL/Direct3D).
        Operation::Min => s.min(d),
        Operation::Max => s.max(d),
    }
}

impl Blend {
    /// The blended color, before conversion to the target format (which
    /// clamps normalized formats).
    pub fn apply(&self, inputs: &Inputs) -> [f32; 4] {
        let mut out = [0.0; 4];
        for (component, value) in out.iter_mut().enumerate() {
            let (op, source, destination) = if component == 3 {
                (self.alpha_op, self.alpha_source, self.alpha_destination)
            } else {
                (self.color_op, self.color_source, self.color_destination)
            };
            *value = combine(
                op,
                inputs.source[component],
                factor(source, inputs, component),
                inputs.destination[component],
                factor(destination, inputs, component),
                inputs.zero_times_anything_is_zero,
            );
        }
        out
    }

    /// Whether any factor reads the second fragment output.
    pub fn dual_source(&self) -> bool {
        [
            self.color_source,
            self.color_destination,
            self.alpha_source,
            self.alpha_destination,
        ]
        .iter()
        .any(|factor| factor.dual_source())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blend(source: Factor, destination: Factor, op: Operation) -> Blend {
        Blend {
            color_op: op,
            color_source: source,
            color_destination: destination,
            alpha_op: op,
            alpha_source: source,
            alpha_destination: destination,
        }
    }

    fn inputs(source: [f32; 4], destination: [f32; 4]) -> Inputs {
        Inputs {
            source,
            source1: [0.0; 4],
            destination,
            constant: [0.25, 0.5, 0.75, 1.0],
            zero_times_anything_is_zero: false,
        }
    }

    #[test]
    fn premultiplied_and_straight_alpha_over() {
        let source = [0.8, 0.4, 0.2, 0.5];
        let destination = [0.0, 0.5, 1.0, 1.0];
        let over = blend(
            Factor::SourceAlpha,
            Factor::OneMinusSourceAlpha,
            Operation::Add,
        )
        .apply(&inputs(source, destination));
        assert_eq!(over, [0.4, 0.45, 0.6, 0.75]);
        let premultiplied = blend(Factor::One, Factor::OneMinusSourceAlpha, Operation::Add)
            .apply(&inputs(source, destination));
        assert_eq!(premultiplied, [0.8, 0.65, 0.7, 1.0]);
    }

    #[test]
    fn subtract_min_max_and_saturate() {
        let i = inputs([0.5, 0.25, 1.0, 0.75], [0.25, 0.5, 0.5, 0.5]);
        let sub = blend(Factor::One, Factor::One, Operation::Subtract).apply(&i);
        assert_eq!(sub, [0.25, -0.25, 0.5, 0.25]);
        let rev = blend(Factor::One, Factor::One, Operation::ReverseSubtract).apply(&i);
        assert_eq!(rev, [-0.25, 0.25, -0.5, -0.25]);
        // Factors do not apply to min/max.
        let min = blend(Factor::Zero, Factor::Zero, Operation::Min).apply(&i);
        assert_eq!(min, [0.25, 0.25, 0.5, 0.5]);
        let max = blend(Factor::Zero, Factor::Zero, Operation::Max).apply(&i);
        assert_eq!(max, [0.5, 0.5, 1.0, 0.75]);
        // min(As, 1 - Ad) = min(0.75, 0.5) for color, 1 for alpha.
        let saturate = blend(Factor::SourceAlphaSaturate, Factor::Zero, Operation::Add).apply(&i);
        assert_eq!(saturate, [0.25, 0.125, 0.5, 0.75]);
    }

    #[test]
    fn both_encodings_and_constants() {
        assert_eq!(Factor::from_raw(0x4303), Factor::from_raw(0x06));
        assert_eq!(Operation::from_raw(0x8006), Operation::from_raw(1));
        assert!(Factor::from_raw(0x0C).is_err());
        let i = inputs([1.0; 4], [0.0; 4]);
        let constant = blend(Factor::ConstantColor, Factor::Zero, Operation::Add).apply(&i);
        assert_eq!(constant, [0.25, 0.5, 0.75, 1.0]);
        let mut dual = i;
        dual.source1 = [0.5; 4];
        let b = blend(Factor::Source1Alpha, Factor::Zero, Operation::Add);
        assert!(b.dual_source());
        assert_eq!(b.apply(&dual), [0.5; 4]);
    }
}

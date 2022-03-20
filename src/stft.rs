/*!

Computes the [short-time fourier transform](https://en.wikipedia.org/wiki/Short-time_Fourier_transform)
on streaming data.

## example

```
use stft::STFT;

// Generate ten seconds of fake audio
let sample_rate: usize = 44100;
let seconds: usize = 10;
let sample_count = sample_rate * seconds;
let all_samples = (0..sample_count).map(|x| x as f64).collect::<Vec<f64>>();

// Initialize the short-time fourier transform
let window_size: usize = 1024;
let step_size: usize = 512;
let mut stft = STFT::new(window_size, step_size).unwrap();

// Iterate over all the samples in chunks of 3000 samples.
// In a real program you would probably read from a stream instead.
for some_samples in (&all_samples[..]).chunks(3000) {
    // Append the samples to the internal ringbuffer of the stft
    stft.append_samples(some_samples);

    // Loop as long as there remain window_size samples in the internal
    // ringbuffer of the stft
    while stft.contains_enough_to_compute() {
        // Compute one column of the stft by
        // taking the first window_size samples of the internal ringbuffer,
        // multiplying them with the window,
        // computing the fast fourier transform,
        // taking half of the symetric complex outputs,
        // computing the norm of the complex outputs and
        // taking the log10
        let spectrogram_column = stft.compute_column();

        // Here's where you would do something with the
        // spectrogram_column...

        // Drop step_size samples from the internal ringbuffer of the stft
        // making a step of size step_size
        stft.move_to_next_column();
    }
}

assert!(!stft.is_empty())

```
*/

use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::num_traits::{Float, Signed, Zero};
use rustfft::{Fft, FftNum, FftPlanner};

use strider::{SliceRing, SliceRingImpl};

pub struct STFT<T>
where
    T: FftNum + FromF64 + Float,
{
    pub window_size: usize,
    pub step_size: usize,
    pub fft: Arc<dyn Fft<T>>,
    pub window: Vec<T>,
    pub sample_ring: SliceRingImpl<T>,
    pub real_input: Vec<T>,
    pub complex_input: Vec<Complex<T>>,
    pub complex_output: Vec<Complex<T>>,
}

impl<T> STFT<T>
where
    T: FftNum + FromF64 + Float,
{
    pub fn new(window_size: usize, step_size: usize) -> Result<Self, String> {
        // TODO: remove dependency on apodize and add additional window types
        let window = apodize::hanning_iter(window_size)
            .map(FromF64::from_f64)
            .collect();

        Self::with_window_vec(window, window_size, step_size)
    }

    pub fn hann(window_size: usize, step_size: usize) -> Result<Self, String> {
        Self::new(window_size, step_size)
    }

    pub fn hamming(window_size: usize, step_size: usize) -> Result<Self, String> {
        let window = apodize::hamming_iter(window_size)
            .map(FromF64::from_f64)
            .collect();

        Self::with_window_vec(window, window_size, step_size)
    }

    pub fn blackman(window_size: usize, step_size: usize) -> Result<Self, String> {
        let window = apodize::blackman_iter(window_size)
            .map(FromF64::from_f64)
            .collect();

        Self::with_window_vec(window, window_size, step_size)
    }

    pub fn nuttall(window_size: usize, step_size: usize) -> Result<Self, String> {
        let window = apodize::nuttall_iter(window_size)
            .map(FromF64::from_f64)
            .collect();

        Self::with_window_vec(window, window_size, step_size)
    }

    pub fn rectangular(window_size: usize, step_size: usize) -> Result<Self, String> {
        Self::with_window_vec(vec![], window_size, step_size)
    }

    pub fn boxcar(window_size: usize, step_size: usize) -> Result<Self, String> {
        Self::rectangular(window_size, step_size)
    }

    pub fn no_window(window_size: usize, step_size: usize) -> Result<Self, String> {
        Self::rectangular(window_size, step_size)
    }

    pub fn output_size(&self) -> usize {
        self.window_size / 2
    }

    pub fn len(&self) -> usize {
        self.sample_ring.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn append_samples(&mut self, input: &[T]) {
        self.sample_ring.push_many_back(input);
    }

    pub fn contains_enough_to_compute(&self) -> bool {
        self.window_size <= self.sample_ring.len()
    }

    /// Computes a column of the spectrogram
    pub fn compute_column(&mut self) -> Result<Vec<T>, String> {
        self.compute_into_complex_output()?;

        Ok(self
            .complex_output
            .iter()
            .map(|x| log10_positive(x.norm()))
            .collect())
    }

    pub fn compute_complex_column(&mut self) -> Result<Vec<Complex<T>>, String> {
        self.compute_into_complex_output()?;

        Ok(self.complex_output.clone())
    }

    pub fn compute_magnitude_column(&mut self) -> Result<Vec<T>, String> {
        self.compute_into_complex_output()?;

        Ok(self.complex_output.iter().map(|x| x.norm()).collect())
    }

    /// Make a step
    /// Drops `self.step_size` samples from the internal buffer `self.sample_ring`.
    pub fn move_to_next_column(&mut self) {
        self.sample_ring.drop_many_front(self.step_size);
    }

    // TODO this should ideally take an iterator and not a vec
    fn with_window_vec(
        window: Vec<T>,
        window_size: usize,
        step_size: usize,
    ) -> Result<Self, String> {
        if !is_power_of_two(window_size) {
            return Err("window size must be a power of two".to_string());
        }
        if step_size <= 0 {
            return Err("step size must be greater than zero".to_string());
        }
        if step_size > window_size {
            return Err("step size must be smaller than or equal to the window size".to_string());
        }

        let inverse = false;
        let mut planner = FftPlanner::new();
        let fft = if inverse {
            planner.plan_fft_inverse(window_size)
        } else {
            planner.plan_fft_forward(window_size)
        };
        Ok(STFT {
            window_size,
            step_size,
            fft,
            sample_ring: SliceRingImpl::new(),
            window,
            real_input: std::iter::repeat(T::zero()).take(window_size).collect(),
            complex_input: std::iter::repeat(Complex::<T>::zero())
                .take(window_size)
                .collect(),
            complex_output: std::iter::repeat(Complex::<T>::zero())
                .take(window_size)
                .collect(),
        })
    }

    fn compute_into_complex_output(&mut self) -> Result<(), String> {
        if !self.contains_enough_to_compute() {
            return Err("not enough data to compute".to_string());
        }

        // Read into real_input
        self.sample_ring.read_many_front(&mut self.real_input[..]);

        // Multiply real_input with window
        for (dst, src) in self.real_input.iter_mut().zip(self.window.iter()) {
            *dst = *dst * *src;
        }

        // Copy windowed real_input as real parts into complex_input
        for (dst, src) in self.complex_input.iter_mut().zip(self.real_input.iter()) {
            dst.re = *src;
        }

        self.complex_output = self.complex_input.clone();
        // Compute fft
        self.fft.process(&mut self.complex_output);
        Ok(())
    }
}

pub trait FromF64 {
    fn from_f64(n: f64) -> Self;
}

impl FromF64 for f64 {
    fn from_f64(n: f64) -> Self {
        n
    }
}

impl FromF64 for f32 {
    fn from_f64(n: f64) -> Self {
        n as f32
    }
}

fn is_power_of_two(n: usize) -> bool {
    n & (n - 1) == 0
}

/// Returns `0` if `log10(value).is_negative()`,
/// otherwise returns `log10(value)`.
/// `log10` turns values in domain `0..1` into values
/// in range `-inf..0`.
/// `log10_positive` turns values in domain `0..1` into `0`.
/// This sets very small values to zero which may not be
/// what you want depending on your application.
fn log10_positive<T: Float + Signed + Zero>(value: T) -> T {
    let log = value.log10();
    if log.is_negative() {
        T::zero()
    } else {
        log
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log10_positive() {
        assert_eq!(log10_positive(-1.), 0.);
        assert_eq!(log10_positive(0.), 0.);
        assert_eq!(log10_positive(1.), 0.);
        assert_eq!(log10_positive(10.), 1.);
        assert_eq!(log10_positive(100.), 2.);
        assert_eq!(log10_positive(1000.), 3.);
    }

    #[test]
    fn test_stft() {
        let mut stft = STFT::new(8, 4).unwrap();
        assert!(!stft.contains_enough_to_compute());
        assert_eq!(stft.output_size(), 4);
        assert_eq!(stft.len(), 0);
        stft.append_samples(&[500., 0., 100.]);
        assert_eq!(stft.len(), 3);
        assert!(!stft.contains_enough_to_compute());
        stft.append_samples(&[500., 0., 100., 0.]);
        assert_eq!(stft.len(), 7);
        assert!(!stft.contains_enough_to_compute());

        stft.append_samples(&[500.]);
        assert!(stft.contains_enough_to_compute());

        let output = stft.compute_column();
        println!("{:?}", output);
    }
}

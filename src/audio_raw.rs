use microflow::buffer::Buffer4D;

pub const AUDIO_LEN: usize = (super::spectrogram::FRAMES - 1) * super::spectrogram::FRAME_STEP + super::spectrogram::FRAME_LENGTH;

pub fn prepare(audio: &[f32]) -> Buffer4D<f32, 1, AUDIO_LEN, 1, 1> {
    let mut buf = Buffer4D::from_element(0.0f32);

    let mut i = 0usize;
    while i < AUDIO_LEN {
        let sample = if i < audio.len() { audio[i] } else { 0.0 };
        buf[(0, i, 0, 0)] = sample;
        i += 1;
    }

    buf
}


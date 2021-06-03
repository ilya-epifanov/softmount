use opencv::{
    prelude::Mat,
    videoio::{
        VideoCapture, VideoCaptureTrait, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT, CAP_PROP_POS_FRAMES,
        CAP_PROP_POS_MSEC,
    },
};
use thiserror::Error;

pub struct VideoSource {
    cap: VideoCapture,
    current_frame: usize,
    frame_count: usize,
    fps: f64,
    frame_duration: f64,
    last_timestamp: f64,
}

#[derive(Debug, Error)]
pub enum VideoSourceError {
    #[error("couldn't get frame count")]
    FrameCount(opencv::Error),
    #[error("coulnd't get FPS")]
    FPS(opencv::Error),
    #[error("coulnd't seek to frame")]
    Seek(opencv::Error),
    #[error("coulnd't read frame")]
    Read(opencv::Error),
    #[error("coulnd't read frame")]
    ReadUnspecified,
    #[error("coulnd't get timestamp")]
    Timestamp(opencv::Error),
}

impl VideoSource {
    pub fn new(cap: VideoCapture) -> Result<Self, VideoSourceError> {
        let frame_count = cap
            .get(CAP_PROP_FRAME_COUNT)
            .map_err(VideoSourceError::FrameCount)?
            .round() as usize;
        let fps = cap.get(CAP_PROP_FPS).map_err(VideoSourceError::FPS)?;
        let frame_duration = 1.0 / fps;

        // cap.set(CAP_PROP_POS_FRAMES, (frame_count - 1) as f64)
        //     .map_err(VideoSourceError::Seek)?;
        // cap.grab().unwrap();
        // let last_timestamp = cap
        //     .get(CAP_PROP_POS_MSEC)
        //     .map_err(VideoSourceError::Timestamp)?
        //     / 1000.0;

        Ok(Self {
            cap,
            current_frame: 0,
            frame_count,
            fps,
            frame_duration,
            last_timestamp: frame_count as f64 * frame_duration,
        })
    }

    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    pub fn fps(&self) -> f64 {
        self.fps
    }

    pub fn frame_duration(&self) -> f64 {
        self.frame_duration
    }

    pub fn last_timestamp(&self) -> f64 {
        self.last_timestamp
    }

    fn seek(&mut self, frame: usize) -> Result<(), VideoSourceError> {
        self.cap
            .set(CAP_PROP_POS_FRAMES, frame as f64)
            .map_err(VideoSourceError::Seek)?;
        Ok(())
    }

    fn next_frame(&mut self, frame: &mut Mat) -> Result<(), VideoSourceError> {
        let success = self.cap.read(frame).map_err(VideoSourceError::Read)?;
        if !success {
            Err(VideoSourceError::ReadUnspecified)
        } else {
            Ok(())
        }
    }

    pub fn get_frame(&mut self, ix: usize, frame: &mut Mat) -> Result<f64, VideoSourceError> {
        if ix != self.current_frame {
            self.seek(ix)?;
        }
        self.next_frame(frame)?;
        let time = self
            .cap
            .get(CAP_PROP_POS_MSEC)
            .map_err(VideoSourceError::Timestamp)?
            / 1000.0;
        Ok(time)
    }
}

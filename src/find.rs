use ffmpeg_sidecar::event::OutputVideoFrame;
use image::{ImageFormat, RgbImage};
use std::path::{Path, PathBuf};
use ffmpeg_sidecar::command::FfmpegCommand;
use std::fs;
use std::ops::Deref;

use crate::Args;
use crate::rife::Rife;
use frame_compare::FrameDifference;

mod tiny_motion_net;
mod frame_compare;
mod backtrackable;

#[derive(Debug, Clone, Copy)]
struct ExponentialMovingAverage {
    pub alpha: f32,
    pub value: Option<f32>
}

impl ExponentialMovingAverage {
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha,
            value: None,
        }
    }

    pub fn commit(&mut self, value: f32) {
        self.value = match self.value {
            Some(prev) => Some(self.alpha * value + (1.0 - self.alpha) * prev),
            None => Some(value),
        };
    }
}

impl Deref for ExponentialMovingAverage {
    type Target = Option<f32>;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

#[derive(Debug)]
pub struct DuplicateChain {
    pub frames: Vec<u32>,
    pub timestamps: Vec<f32>,
    pub frames_dir: PathBuf,
}

#[derive(Default)]
enum FindChainKind {
    #[default]
    Empty,
    FindDuplicate,
    CompensateMotion {
        n_additional: usize
    }
}

#[derive(Default)]
struct FindChainState {
    frames: Vec<OutputVideoFrame>,
    kind: FindChainKind,
}

impl FindChainState {

    pub fn push(&mut self, frame: OutputVideoFrame) {
        self.frames.push(frame);
        match &mut self.kind {
            FindChainKind::Empty => self.kind = FindChainKind::FindDuplicate,
            FindChainKind::FindDuplicate => {},
            FindChainKind::CompensateMotion { n_additional} => *n_additional += 1,
        }
    }

    pub fn clear(&mut self) {
        self.frames.clear();
        self.kind = FindChainKind::Empty;
    }

    pub fn last(&self) -> Option<&OutputVideoFrame> {
        self.frames.last()
    }

    pub fn iter(&mut self) -> impl Iterator<Item = &OutputVideoFrame> {
        self.frames.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut OutputVideoFrame> {
        self.frames.iter_mut()
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn remove(&mut self, i: u32) -> OutputVideoFrame {
        let frame = self.frames.remove(i as usize);
        if self.frames.is_empty() {
            self.kind = FindChainKind::Empty;
        }
        frame
    }

}


fn frame_to_image(frame: OutputVideoFrame) -> RgbImage {
    image::ImageBuffer::from_vec(frame.width, frame.height, frame.data).unwrap()
}

pub fn find_duplicates(args: &Args, rife: &mut Rife) {
    let iter = FfmpegCommand::new()
        .input(args.input_path.display().to_string())
        .rawvideo()
        .spawn().expect("Ffmpeg should spawn")
        .iter().expect("Should be able to get Ffmpeg event iterator");

    let s = std::time::Instant::now();

    let mut current_chain = FindChainState::default();
    let mut last_diff = 0.0;
    let mut avg_hash = ExponentialMovingAverage::new(args.diff_mean_alpha);
    let mut recent_motion = ExponentialMovingAverage::new(args.recent_motion_mean_alpha); // "fast" averaged last motion
    let mut slow_avg_motion = ExponentialMovingAverage::new(args.slow_motion_mean_alpha);
    let mut chain_motion = 0.0;
    let mut chain_i = 0;

    for frame in iter.filter_frames() {
        let diff = match current_chain.last() {
            None => FrameDifference::INFINITY,
            Some(previous) => frame_compare::compare_frames(previous, &frame, args),
        };
        if !diff.is_normal() {
            current_chain.push(frame);
            continue;
        }

        chain_motion += diff.motion_estimate;

        let avg_diff = avg_hash.unwrap_or(0.0);
        let threshold = (avg_diff * args.mul_dup_threshold).min(args.max_dup_threshold);

        //println!("#{} - #{} diff: {:.4}, motion: {:.4}", frame.frame_num - 1, frame.frame_num, diff.hash_distance, diff.motion_estimate);
        if diff.hash_distance > threshold {
            avg_hash.commit(diff.hash_distance);
            slow_avg_motion.commit(diff.motion_estimate);

            let chain_result = process_new_chain(ProcessChainData {
                chain_i,
                chain_motion,
                chain: &mut current_chain,
                end_frame: &frame,
                slow_avg_motion: &slow_avg_motion,
                recent_motion: &recent_motion,
            }, args);

            match chain_result {
                ProcessChainResult::Fail => {
                    chain_motion = 0.0;
                    recent_motion.commit(diff.motion_estimate);
                }
                ProcessChainResult::Continue => {} // Compensate motion -> Try next
                ProcessChainResult::Ok(chain) => {
                    println!(
                        "Duplicates {} found #{}-#{}, length: {}, diff_ema: {:0.4}, diff: {:0.4}, chain_motion: {:0.4} at {:.3}s",
                        chain_i, chain.frames[0], chain.frames.last().unwrap(), chain.frames.len(), avg_diff, last_diff, chain_motion, chain.timestamps[0]
                    );

                    chain_i += 1;
                    let patch_dir = format!("tmp/patch_{}", chain_i);
                    fs::create_dir_all(&patch_dir).expect("Creating patch dir should not fail");
                    rife.generate_in_betweens(
                        chain,
                        &patch_dir,
                    );
                    chain_motion = 0.0;
                    recent_motion.commit(diff.motion_estimate);
                }
            }
        }
        let chain_length = current_chain.len();
        if chain_length > args.max_duplicates {
            // Free up memory from too long chains using dummy frames
            current_chain.iter_mut().for_each(|f| f.data.clear());
        }

        current_chain.push(frame);
        last_diff = diff.hash_distance;
    }

    println!("Finding Duplicates Took: {}s", s.elapsed().as_secs_f32());
}


struct ProcessChainData<'a> {
    chain_i: usize,
    chain_motion: f32,
    chain: &'a mut FindChainState,
    end_frame: &'a OutputVideoFrame,
    slow_avg_motion: &'a ExponentialMovingAverage,
    recent_motion: &'a ExponentialMovingAverage
}


enum ProcessChainResult {
    Fail,
    Continue,
    Ok(DuplicateChain)
}


const N_ACTIVATE_MOTION_COMPENSATE: usize = 3;

fn process_new_chain(data: ProcessChainData, args: &Args) -> ProcessChainResult {
    let len = data.chain.len();
    // Check bounds
    if args.min_duplicates > len || len > args.max_duplicates {
        data.chain.clear();
        return ProcessChainResult::Fail;
    }

    if len >= N_ACTIVATE_MOTION_COMPENSATE {
        // Check motion compensation
        let recent_motion = data.recent_motion.unwrap_or(0.0);
        let required_motion = recent_motion * (len - 1) as f32 * args.motion_compensate_threshold;
        if data.chain_motion < required_motion {
            data.chain.kind = FindChainKind::CompensateMotion { n_additional: 0};
            return ProcessChainResult::Continue;
        }
        let slow_avg_motion = data.slow_avg_motion.unwrap_or(0.0);
        if data.chain_motion > slow_avg_motion * args.max_motion_mul {
            data.chain.clear();
            return ProcessChainResult::Fail;
        }
    }

    let frames: Vec<_> = data.chain.iter().map(|f| f.frame_num).collect();
    let timestamps: Vec<_> = data.chain.iter().map(|f| f.timestamp).collect();

    let start_image = frame_to_image(data.chain.remove(0));
    let end_image = frame_to_image(data.end_frame.clone());

    let frames_path = Path::new("tmp/frames");
    let dir = frames_path.join(data.chain_i.to_string());
    fs::create_dir_all(&dir).expect("Creating frames directory should not fail");

    let start_frame_path = dir.join("0.webp").to_string_lossy().to_string();
    let end_frame_path = dir.join("1.webp").to_string_lossy().to_string();
    start_image.save_with_format(&start_frame_path, ImageFormat::WebP).expect("Writing start image should not fail");
    end_image.save_with_format(&end_frame_path, ImageFormat::WebP).expect("Writing start image should not fail");
    data.chain.clear();

    let chain = DuplicateChain {
        frames,
        timestamps,
        frames_dir: dir
    };
    ProcessChainResult::Ok(chain)
}

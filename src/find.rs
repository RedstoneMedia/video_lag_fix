use std::fmt::{Debug, Formatter};
use ffmpeg_sidecar::event::OutputVideoFrame;
use image::{ImageFormat, RgbImage};
use std::path::{Path, PathBuf};
use ffmpeg_sidecar::command::FfmpegCommand;
use std::fs;
use std::ops::Deref;
use log::{debug, info, warn};
use crate::Args;
use crate::rife::Rife;
use frame_compare::FrameDifference;
use crate::find::backtrackable::{BacktrackCtx, Backtrackable};
use crate::find::frame_compare::{preprocess_frame, PreprocessedFrame};

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

#[derive(Copy, Clone, Debug)]
enum FindState {
    FindDuplicate,
    CompensateMotion {
        n_additional: usize
    },
    FailedCompensate
}

impl PartialEq for FindState {
    fn eq(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (FindState::FindDuplicate, FindState::FindDuplicate)
            | (FindState::CompensateMotion { .. }, FindState::CompensateMotion { .. })
            | (FindState::FailedCompensate, FindState::FailedCompensate)
        )
    }
}

fn frame_to_image(frame: OutputVideoFrame) -> RgbImage {
    image::ImageBuffer::from_vec(frame.width, frame.height, frame.data).unwrap()
}

#[derive(Copy, Clone)]
struct IterVars {
    chain_i: usize,
    chain_motion: f32,
    last_diff: f32,
    avg_hash: ExponentialMovingAverage,
    recent_motion: ExponentialMovingAverage,
    slow_avg_motion: ExponentialMovingAverage,
}

impl Debug for IterVars {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "IterVars {{i: {}}}", self.chain_i)
    }
}

impl IterVars {

    pub fn from_args(args: &Args) -> Self {
        Self {
            chain_i: 0,
            chain_motion: 0.0,
            last_diff: 0.0,
            avg_hash: ExponentialMovingAverage::new(args.diff_mean_alpha),
            recent_motion: ExponentialMovingAverage::new(args.recent_motion_mean_alpha),
            slow_avg_motion: ExponentialMovingAverage::new(args.slow_motion_mean_alpha),
        }
    }

}


pub fn find_duplicates(
    input_path: impl AsRef<Path>,
    mut rife: Option<&mut Rife>,
    args: &Args
) {
    let input_path = input_path.as_ref();
    let iter = FfmpegCommand::new()
        .input(input_path.display().to_string())
        .rawvideo()
        .spawn().expect("Ffmpeg should spawn")
        .iter().expect("Should be able to get Ffmpeg event iterator");

    let s = std::time::Instant::now();

    let mut state = FindState::FindDuplicate;
    let vars = IterVars::from_args(args);
    let max_history = args.max_duplicates * 2 + 1; // Kind of makes sense if you want the full history while backtracking. +1 to detect too long
    let mut last_found_frame = 0u32;
    let preprocessed_frames_iter = iter.filter_frames().map(|frame| {
        let preprocessed_frame = preprocess_frame(&frame, args);
        (frame, preprocessed_frame)
    });
    let frames_backtrack = Backtrackable::new(preprocessed_frames_iter, vars, max_history);
    frames_backtrack.for_each(|vars, (frame, preprocessed_frame), iter_ctx| {
        let diff = match iter_ctx.last() {
            None => FrameDifference::INFINITY,
            Some((_, previous)) => frame_compare::compare_frames(previous, preprocessed_frame),
        };
        if !diff.is_finite() {
            warn!("got non-finite diff: {:?}", diff);
            state = FindState::FindDuplicate;
            iter_ctx.clear();
            return;
        }
        vars.chain_motion += diff.motion_estimate;

        let check_result = check_chain(ProcessChainData {
            vars,
            iter_ctx,
            state,
            diff,
            end_frame: frame,
        }, args);
        match (check_result, state) {
            (CheckChainResult::FailDuplicate, _) => {
                debug!("Found duplicate at #{}", frame.frame_num);
            }, // skip
            // Lost cause, move on
            (CheckChainResult::FailShort | CheckChainResult::FailLong | CheckChainResult::FailTooMuchMotion, FindState::FindDuplicate | FindState::FailedCompensate)
            | (CheckChainResult::FailShort, FindState::CompensateMotion {..})
            | (CheckChainResult::CompensationRequired, FindState::FailedCompensate) // Give up
            => {
                if check_result != CheckChainResult::FailShort {
                    let start_frame = frame.frame_num as usize - iter_ctx.len();
                    debug!("Skipping duplicate chain #{}-#{}: #{:?}, state: {:?}", start_frame, frame.frame_num, check_result, state);
                }

                vars.chain_motion = 0.0;
                vars.recent_motion.commit(diff.motion_estimate);
                state = FindState::FindDuplicate;
                iter_ctx.clear();
            },
            // Try again
            (CheckChainResult::FailTooMuchMotion | CheckChainResult::FailLong, FindState::CompensateMotion { n_additional }) => {
                state = FindState::FailedCompensate;
                iter_ctx.backtrack(n_additional);
            },
            // Check further
            (CheckChainResult::CompensationRequired, FindState::FindDuplicate) => {
                state = FindState::CompensateMotion { n_additional: 1 }; // Compensate motion -> Try next
            },
            (CheckChainResult::CompensationRequired, FindState::CompensateMotion {n_additional}) => {
                state = FindState::CompensateMotion { n_additional: n_additional + 1};
            },
            (CheckChainResult::Ok, _) => {
                let chain = create_new_chain(ProcessChainData {
                    vars,
                    iter_ctx,
                    state,
                    end_frame: frame,
                    diff,
                });
                let avg_diff = vars.avg_hash.unwrap_or(0.0);
                info!(
                    "Duplicate chain {} found #{}-#{}, length: {}, diff_ema: {:0.4}, diff: {:0.4}, chain_motion: {:0.4}, state: {:?}, at {:.3}s",
                    vars.chain_i, chain.frames[0], chain.frames.last().unwrap(), chain.frames.len(), avg_diff, vars.last_diff, vars.chain_motion, state, chain.timestamps[0]
                );

                assert!(last_found_frame < chain.frames[0], "Sequential sanity check {} < {}", last_found_frame, chain.frames[0]);
                last_found_frame = chain.frames[0];

                vars.chain_i += 1;
                vars.chain_motion = 0.0;
                vars.recent_motion.commit(diff.motion_estimate);

                if let Some(rife) = &mut rife {
                    let patch_dir = format!("tmp/patch_{}", vars.chain_i);
                    fs::create_dir_all(&patch_dir).expect("Creating patch dir should not fail");
                    rife.generate_in_betweens(
                        chain,
                        &patch_dir,
                    );
                }

                state = FindState::FindDuplicate;
                iter_ctx.clear();
            }
        }

        if check_result != CheckChainResult::FailDuplicate {
            vars.avg_hash.commit(diff.hash_distance);
            vars.slow_avg_motion.commit(diff.motion_estimate);
        }

        vars.last_diff = diff.hash_distance;
    });

    info!("Finding Duplicates Took: {}s", s.elapsed().as_secs_f32());
}


struct ProcessChainData<'a, 'b> {
    vars: &'a IterVars,
    diff: FrameDifference,
    state: FindState,
    iter_ctx: &'a mut BacktrackCtx<'b, IterVars, (OutputVideoFrame, PreprocessedFrame)>,
    end_frame: &'a OutputVideoFrame
}


#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum CheckChainResult {
    FailDuplicate,
    FailShort,
    FailLong,
    FailTooMuchMotion,
    CompensationRequired,
    Ok
}

fn check_chain(data: ProcessChainData, args: &Args) -> CheckChainResult {
    let vars = data.vars;
    let state = data.state;
    let diff = data.diff;
    let iter_ctx = data.iter_ctx;

    // Check too long
    let len = iter_ctx.len();
    if len > args.max_duplicates + 1 {
        return CheckChainResult::FailLong;
    }
    // Check for duplicate frame
    let avg_diff = vars.avg_hash.unwrap_or(0.0);
    let threshold = (avg_diff * args.mul_dup_threshold).min(args.max_dup_threshold);
    if diff.hash_distance < threshold {
        return CheckChainResult::FailDuplicate;
    }
    // Check short after duplicate (Otherwise chains can't start at all)
    if args.min_duplicates > len {
        return CheckChainResult::FailShort;
    }
    // Check motion compensation
    if len > args.motion_compensate_start {
        let recent_motion = vars.recent_motion.unwrap_or(0.0);
        // Ask for less compensation if we failed it before
        // That way there will at least be slightly flawed (slow) interpolation rather than a freeze-frame.
        let motion_compensate_threshold = if state == FindState::FailedCompensate {
            args.motion_compensate_threshold * args.mul_motion_compensate_threshold_retry
        } else {
            args.motion_compensate_threshold
        };
        let required_motion = recent_motion * (len - 1) as f32 * motion_compensate_threshold;
        if vars.chain_motion < required_motion {
            return CheckChainResult::CompensationRequired;
        }
    }
    // Check too much motion
    let slow_avg_motion = vars.slow_avg_motion.unwrap_or(0.0);
    if vars.chain_motion > slow_avg_motion * args.max_motion_mul {
        return CheckChainResult::FailTooMuchMotion;
    }
    CheckChainResult::Ok
}

fn create_new_chain(data: ProcessChainData) -> DuplicateChain {
    let vars = data.vars;
    let iter_ctx = data.iter_ctx;

    // Construct Duplicate Chain
    let frames: Vec<_> = iter_ctx.iter().map(|(f, _)| f.frame_num).collect();
    let timestamps: Vec<_> = iter_ctx.iter().map(|(f, _)| f.timestamp).collect();

    let mut chain = iter_ctx.clear_and_drain(); // clear history side effect to avoid cloning large frames
    let start_image = frame_to_image(chain.pop_front().expect("Chain should have frame").0);
    let end_image = frame_to_image(data.end_frame.clone());

    let frames_path = Path::new("tmp/frames");
    let dir = frames_path.join(vars.chain_i.to_string());
    fs::create_dir_all(&dir).expect("Creating frames directory should not fail");

    let start_frame_path = dir.join("0.webp").to_string_lossy().to_string();
    let end_frame_path = dir.join("1.webp").to_string_lossy().to_string();
    start_image.save_with_format(&start_frame_path, ImageFormat::WebP).expect("Writing start image should not fail");
    end_image.save_with_format(&end_frame_path, ImageFormat::WebP).expect("Writing start image should not fail");

    DuplicateChain {
        frames,
        timestamps,
        frames_dir: dir
    }
}

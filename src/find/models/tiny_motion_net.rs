use std::sync::{Arc, Mutex};
use fast_image_resize::images::Image;
use once_cell::sync::Lazy;
use ort::session::{Session};
use ort::inputs;
use ort::value::Tensor;
use crate::find::models::{create_model_session, INPUT_SIZE};

const MODEL_PATH: &str = "MotionPredictor/models/v8.onnx";
const THREADS: usize = 3;

static SESSION: Lazy<Arc<Mutex<Session>>> = Lazy::new(|| {
    let session = create_model_session(MODEL_PATH, THREADS);
    Arc::new(Mutex::new(session))
});

/// Predicts the motion between two images
///
/// [net::preprocess_image] is expected to already be applied to the images
pub fn predict_motion(img_a: &Image, img_b: &Image) -> f32 {
    // Diff
    let diff = img_a.buffer().iter().zip(img_b.buffer())
        .map(|(a, b)| *a as f32 / 255.0 - *b as f32 / 255.0).collect::<Vec<_>>();
    let input = Tensor::from_array(([1, 1, INPUT_SIZE.1, INPUT_SIZE.0], diff)).expect("Difference shape mismatch");
    // Predict
    let mut session = SESSION.lock().expect("Could not lock ONNX session");
    let preds = session.run(inputs!["input" => input]).expect("Could not run model");
    let (_, motion): (_, &[f32]) = preds.get("output").expect("Model did not have output")
        .try_extract_tensor().expect("Could not get f32 model output");
    motion[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx() {
        let mut model = create_model_session("../../../MotionPredictor/models/v8.onnx", 2);

        let t = std::time::Instant::now();
        let n = 1000;
        for _ in 0..n {
            let image = Tensor::from_array(([1, 1, 180, 320], vec![0.0f32; 1*180*320])).unwrap();
            let preds = model.run(inputs!["input" => image]).unwrap();
            let (_, _): (_, &[f32]) = preds.get("output").unwrap().try_extract_tensor().unwrap();
        }
        let secs = t.elapsed().as_secs_f64() / n as f64;
        println!("{:.01} MP/S", 1.0 / secs);
    }

}
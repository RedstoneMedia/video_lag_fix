use std::sync::{Arc, Mutex};
use fast_image_resize::images::Image;
use once_cell::sync::Lazy;
use ort::inputs;
use ort::session::Session;
use ort::value::Tensor;
use crate::find::models::{create_model_session, INPUT_SIZE};

const MODEL_PATH: &str = "DuplicateDetect/models/conv-v3.onnx";
const THREADS: usize = 2;

static SESSION: Lazy<Arc<Mutex<Session>>> = Lazy::new(|| {
    let session = create_model_session(MODEL_PATH, THREADS);
    Arc::new(Mutex::new(session))
});

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}


/// Predicts the probability that two images are distinct
///
/// [net::preprocess_image] is expected to already be applied to the images
pub fn predict_distinct(img_a: &Image, img_b: &Image) -> f32 {
    let a_norm = img_a.buffer().iter().map(|a| *a as f32 / 255.0);
    let b_norm = img_b.buffer().iter().map(|b| *b as f32 / 255.0);
    let stacked = a_norm.chain(b_norm).collect::<Vec<_>>();
    let input = Tensor::from_array(([1, 2, INPUT_SIZE.1, INPUT_SIZE.0], stacked)).expect("Difference shape mismatch");
    // Predict
    let mut session = SESSION.lock().expect("Could not lock ONNX session");
    let preds = session.run(inputs!["input" => input]).expect("Could not run model");
    let (_, raw_output): (_, &[f32]) = preds.get("output").expect("Model did not have output")
        .try_extract_tensor().expect("Could not get f32 model output");
    let raw_logit = raw_output[0];
    sigmoid(raw_logit)
}

#[cfg(test)]
mod tests {
    use crate::find::models::create_model_session;
    use super::*;

    #[test]
    fn test_onnx() {
        let mut model = create_model_session(MODEL_PATH, 2);

        let t = std::time::Instant::now();
        let n = 1000;
        for _ in 0..n {
            let image = Tensor::from_array(
                ([1, 2, INPUT_SIZE.1, INPUT_SIZE.0],
                 vec![0.0f32; 2 * INPUT_SIZE.1 * INPUT_SIZE.0])
            ).unwrap();
            let preds = model.run(inputs!["input" => image]).unwrap();
            let (_, _): (_, &[f32]) = preds.get("output").unwrap().try_extract_tensor().unwrap();
        }
        let secs = t.elapsed().as_secs_f64() / n as f64;
        println!("{:.01} DP/S", 1.0 / secs);
    }

}
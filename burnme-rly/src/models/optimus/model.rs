//! Optimus iTransformer model using Candle/itransformer

use super::config::OptimusConfig;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use itransformer::ITransformer;

/// Optimus model wrapper around iTransformer
pub struct OptimusModel {
    inner: ITransformer,
    config: OptimusConfig,
}

impl OptimusModel {
    /// Create new Optimus model from config
    pub fn new(config: &OptimusConfig, device: &Device) -> candle_core::Result<Self> {
        // Create varmap for parameters
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, device);

        // Create iTransformer
        let inner = ITransformer::new(
            vb,
            config.num_variates,
            config.lookback_len,
            config.n_layers,
            config.d_model,
            Some(1),                               // num_tokens_per_variate
            vec![config.pred_len],                 // prediction lengths
            Some(config.d_model / config.n_heads), // dim_head
            Some(config.n_heads),
            Some(config.dropout), // attn_drop_p
            None,                 // ff_mult
            Some(config.dropout), // ff_drop_p
            None,                 // num_mem_tokens
            Some(config.use_revin),
            None,  // revin_affine
            false, // flash_attn
            device,
        )?;

        Ok(Self {
            inner,
            config: config.clone(),
        })
    }

    /// Forward pass - returns prediction for the configured horizon
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let result = self.inner.forward(x, None, false)?;

        // iTransformer returns Either<Vec<(usize, Tensor)>, f64>
        // Extract the first prediction tensor from the Vec variant
        match result {
            either::Either::Left(predictions) => {
                // Return the first prediction horizon's tensor
                predictions
                    .into_iter()
                    .next()
                    .map(|(_, tensor)| tensor)
                    .ok_or_else(|| candle_core::Error::Msg("No predictions returned".to_string()))
            }
            either::Either::Right(_) => Err(candle_core::Error::Msg(
                "Unexpected scalar result from iTransformer".to_string(),
            )),
        }
    }

    /// Get config
    pub fn config(&self) -> &OptimusConfig {
        &self.config
    }
}

# Frozen v4.2 runtime asset

Upload `v42_runtime_bundle.zip` into this folder as `backend/assets/v42_runtime_bundle.zip`.

The bundle contains the frozen trained-through-2025 v4.2 model artifact, compact frozen 2018-2025 feature histories, the model modules required for inference, and the automated refresh script. The refresh pipeline updates live inputs only. It does not retrain or tune v4.2 on 2026 outcomes.

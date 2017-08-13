gcloud ml-engine jobs submit training fcn01 \
    --job-dir gs://tfjobs \
    --staging-bucket gs://tfstaging \
    --package-path trainer \
    --module-name trainer.main \
    --region us-central1

cd ../
python -m scripts.fit_model \
          --level=1 \
          --lr=0.002 \
          --zoom_level=2 \
          --model_type="multi" \
          --gpu_memory=7000

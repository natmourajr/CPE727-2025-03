#!/bin/bash
# View MLflow runs without starting UI

echo "=== MLFLOW EXPERIMENTS ===="
echo ""

for exp_dir in results/mlruns/*/; do
    if [ -f "$exp_dir/meta.yaml" ]; then
        exp_name=$(grep "^name:" "$exp_dir/meta.yaml" | cut -d' ' -f2-)
        exp_id=$(basename "$exp_dir")
        
        if [ -n "$exp_name" ]; then
            run_count=$(ls -1 "$exp_dir" | grep -v -E "^(meta.yaml|models|outputs|\\.DS_Store|\\.trash)$" | wc -l | tr -d ' ')
            
            echo "📊 $exp_name"
            echo "   ID: $exp_id"
            echo "   Runs: $run_count"
            echo ""
            
            # List recent runs
            if [ "$run_count" -gt 0 ]; then
                echo "   Recent runs:"
                for run_dir in "$exp_dir"*/; do
                    if [ -f "$run_dir/meta.yaml" ]; then
                        run_name=$(grep "^run_name:" "$run_dir/meta.yaml" | cut -d' ' -f2-)
                        if [ -n "$run_name" ]; then
                            echo "   - $run_name"
                        fi
                    fi
                done
                echo ""
            fi
        fi
    fi
done

echo "=== HOW TO VIEW IN MLFLOW UI ===="
echo "1. Run: uv run mlflow ui --backend-store-uri file://$(pwd)/results/mlruns"
echo "2. Open: http://127.0.0.1:5000"
echo "3. Click experiment dropdown (top-left)"
echo "4. Select: fashion-mnist-pca-experiments or hierarchical-classifier-experiments"
echo ""

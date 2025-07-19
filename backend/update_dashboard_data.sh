#!/bin/bash
# Update dashboard data by generating new activation data and copying to frontend

echo "Updating LoRA activations dashboard data..."
echo "=========================================="

# Default values
NUM_EXAMPLES=50

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-examples)
            NUM_EXAMPLES="$2"
            shift 2
            ;;
        *)
            # Pass through other arguments to the Python script
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Run the Python script
python generate_activations_data.py --num-examples "$NUM_EXAMPLES" $EXTRA_ARGS

if [ $? -eq 0 ]; then
    echo ""
    echo "Dashboard data updated successfully!"
    echo "Run 'npm run dev' in the frontend directory to start the dashboard."
else
    echo ""
    echo "Error: Failed to generate activation data."
    exit 1
fi
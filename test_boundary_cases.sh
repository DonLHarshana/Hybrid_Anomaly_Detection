#!/bin/bash
# 5 Boundary Case Test Runner
set -e
cd "$(dirname "$0")"

echo "=========================================="
echo "5 BOUNDARY CASE TEST SUITE"
echo "=========================================="

# Install Trivy if needed
if [ ! -f "./bin/trivy" ]; then
    echo "Installing Trivy..."
    curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh
fi

mkdir -p test_results

run_scenario() {
    local scenario_num=$1
    local scenario_name=$2
    local inject_profile=$3
    local expected_decision=$4
    
    echo ""
    echo "=========================================="
    echo "SCENARIO $scenario_num: $scenario_name"
    echo "Expected: $expected_decision"
    echo "=========================================="
    
    rm -rf datasets/payment_set_test_$scenario_num trivy_out ml_out
    mkdir -p trivy_out ml_out
    
    # Generate payment set
    INJECT_PROFILE=$inject_profile python3 trivy/make_payment_set_trivy.py \
        --id test_$scenario_num \
        --template trivy/payment_set_template
    
    # Trivy scan
    ./bin/trivy fs --scanners secret --format json \
        -o trivy_out/scan.json datasets/payment_set_test_$scenario_num || true
    
    # Score Trivy
    python3 trivy/score_trivy.py \
        --scan trivy_out/scan.json \
        --gt-csv datasets/payment_set_test_$scenario_num/ground_truth/secrets.csv \
        --out trivy_out/trivy_metrics.json
    
    # ML Evaluate
    python3 ml/src/evaluate.py
    
    # Decision Gate
    python3 ml/src/decision_gate.py || true
    
    # Save results
    cp ml_out/gate_out.json test_results/scenario_${scenario_num}_${scenario_name}.json
    
    echo ""
    echo "Result:"
    cat ml_out/gate_out.json | python3 -m json.tool
    echo ""
}

# Run all 4 scenarios
run_scenario 1 "clean" "clean" "ACCEPT"
run_scenario 2 "low_risk" "low" "ACCEPT"
run_scenario 3 "medium_risk" "medium" "HOLD"
run_scenario 4 "high_risk" "high" "REJECT"

echo ""
echo "=========================================="
echo "TEST SUITE COMPLETE"
echo "=========================================="
echo ""
echo "Results saved to test_results/"
echo ""
echo "Summary:"
for file in test_results/scenario_*.json; do
    if [ -f "$file" ]; then
        name=$(basename "$file" .json)
        decision=$(grep -o '"decision": "[^"]*"' "$file" | cut -d'"' -f4)
        score=$(grep -o '"fusion_score": [0-9.-]*' "$file" | awk '{print $2}')
        risk=$(grep -o '"trivy_risk": "[^"]*"' "$file" | cut -d'"' -f4)
        echo "  $name: $decision (risk=$risk, score=$score)"
    fi
done
echo ""

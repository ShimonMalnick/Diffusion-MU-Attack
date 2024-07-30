methods=('EraseDiff' 'ESD' 'FMN' 'Salun' 'Scissorhands' 'SPM' 'UCE')
for method in "${methods[@]}"; do
    echo "Inverting $method ...."
    python concept-ablation-diffusers/attack_tamed_model_variable_nudity.py --method $method
done

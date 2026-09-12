import argparse
import json
import os


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser(description="Apply the frozen PC-BIND v9 fold-1 gate.")
    parser.add_argument("--control-summary", required=True)
    parser.add_argument("--treatment-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    control = load_json(args.control_summary)
    treatment = load_json(args.treatment_summary)
    control_true = control["performance"]["true"]
    treatment_true = treatment["performance"]["true"]
    treatment_differences = treatment["performance"]["aupr_differences"]
    residual_correlation = treatment["pathway_utilization"]["correlation"][
        "true_vs_mismatch_partner_contribution"
    ]
    values = {
        "true_minus_mismatch_aupr": treatment_differences["true_minus_mismatch"],
        "true_aupr_change_vs_control": (
            treatment_true["auc_pr"] - control_true["auc_pr"]
        ),
        "true_mcc_change_vs_control": treatment_true["mcc"] - control_true["mcc"],
        "true_minus_intrinsic_aupr": treatment_differences["true_minus_intrinsic"],
        "true_vs_mismatch_residual_correlation": residual_correlation,
    }
    criteria = {
        "true_minus_mismatch_aupr_at_least_0.005": values[
            "true_minus_mismatch_aupr"
        ] >= 0.005,
        "true_aupr_change_at_least_minus_0.003": values[
            "true_aupr_change_vs_control"
        ] >= -0.003,
        "true_mcc_change_at_least_minus_0.003": values[
            "true_mcc_change_vs_control"
        ] >= -0.003,
        "true_minus_intrinsic_aupr_positive": values[
            "true_minus_intrinsic_aupr"
        ] > 0.0,
        "residual_correlation_at_most_0.95": residual_correlation <= 0.95,
    }
    output = {
        "protocol": "pcbind_v9_protocol.md",
        "control_summary": os.path.abspath(args.control_summary),
        "treatment_summary": os.path.abspath(args.treatment_summary),
        "values": values,
        "criteria": criteria,
        "decision": "CONTINUE" if all(criteria.values()) else "STOP",
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")

    print("# PC-BIND v9 Frozen Gate")
    for name, passed in criteria.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    print(f"Decision: {output['decision']}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

"""
Pipeline entry point: run the full solution end-to-end.

Usage:
    python -m src.run_pipeline
"""
import sys
import time

from .config  import DATA_DIR, OUTPUT_DIR
from .data    import load_data
from .features import build_features
from .train   import cross_validate, train_final
from .predict import predict_test, build_submission


def main():
    print('=' * 70)
    print('GeoAI Aquaculture Pond Identification — Rank 6 Solution')
    print('=' * 70)

    # 1. Load + ETL
    print('\n[1/4] Loading data...')
    t0 = time.time()
    train, test, ss = load_data()
    print(f'  Train: {train.shape}  Test: {test.shape}  SS: {ss.shape}')
    print(f'  Positive rate in train: {train["label"].mean():.4f}')
    print(f'  Elapsed: {time.time()-t0:.1f}s')

    # 2. Feature schema (run on 0 rows to discover columns without computing)
    feats = build_features(train.iloc[:0]).columns.tolist()
    print(f'\n[2/4] Features: {len(feats)}')

    # 3. Cross-validation (verification)
    print('\n[3/4] Running 5-fold cross-validation...')
    cross_validate(train, feats)

    # 4. Final training + inference + submission
    print('\n[4/4] Training final model + generating submission...')
    t0 = time.time()
    ml_f, mc_f = train_final(train, feats)
    probs = predict_test(ml_f, mc_f, test, feats)

    output_path = str(OUTPUT_DIR / 'Submission_V30_prevalence_matched.csv')
    sub = build_submission(test, probs, ss.columns.tolist(), output_path)

    print(f'\n{"=" * 70}')
    print(f'Done. Submission saved to: {output_path}')
    print(f'Expected scores (from rank-6 submission):')
    print(f'  Public:  0.922151053')
    print(f'  Private: 0.946981266')
    print(f'  AUC:     0.970627769')
    print(f'  F1:      0.931216931')
    print(f'{"=" * 70}')


if __name__ == '__main__':
    main()

"""
Data Verification Script for Service Request Dataset
===================================================
Comprehensive verification of all 50,000+ records
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

def verify_data_integrity():
    print("🔍 STARTING COMPREHENSIVE DATA VERIFICATION")
    print("=" * 60)

    # Load data
    print("📂 Loading CSV data...")
    df = pd.read_csv('scramble-data.csv')
    print(f"✅ Loaded {len(df):,} records")

    # Basic structure verification
    print("\n📊 BASIC STRUCTURE VERIFICATION")
    print("-" * 40)
    expected_columns = [
        'service_request_id', 'user_id', 'service_id', 'name', 'org_name',
        'payment_service', 'status', 'finish_time', 'response', 'created_at',
        'first_name', 'middle_name', 'last_name', 'forth_name', 'phone_num',
        'birth_date', 'nodes'
    ]

    print(f"Expected columns: {len(expected_columns)}")
    print(f"Actual columns: {len(df.columns)}")
    print(f"Columns match: {set(expected_columns) == set(df.columns)}")

    # Data types verification
    print("\n🔢 DATA TYPES VERIFICATION")
    print("-" * 40)
    print(df.dtypes)

    # Status distribution verification
    print("\n📈 STATUS DISTRIBUTION VERIFICATION")
    print("-" * 40)
    status_counts = df['status'].value_counts()
    print("Status distribution:")
    for status, count in status_counts.items():
        print(f"  {status}: {count:,} ({count/len(df)*100:.2f}%)")

    # Organization verification
    print("\n🏢 ORGANIZATION VERIFICATION")
    print("-" * 40)
    org_counts = df['org_name'].value_counts()
    print(f"Total unique organizations: {len(org_counts)}")
    print("Top 10 organizations:")
    for i, (org, count) in enumerate(org_counts.head(10).items(), 1):
        print(f"  {i}. {org}: {count:,}")

    # Date range verification
    print("\n📅 DATE RANGE VERIFICATION")
    print("-" * 40)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['finish_time'] = pd.to_datetime(df['finish_time'], errors='coerce')

    print(f"Created date range: {df['created_at'].min()} to {df['created_at'].max()}")
    print(f"Finish date range: {df['finish_time'].min()} to {df['finish_time'].max()}")

    # Payment service verification
    print("\n💳 PAYMENT SERVICE VERIFICATION")
    print("-" * 40)
    payment_counts = df['payment_service'].value_counts()
    print("Payment service distribution:")
    for payment, count in payment_counts.items():
        print(f"  {payment}: {count:,} ({count/len(df)*100:.2f}%)")

    # Processing time calculation verification
    print("\n⏱️  PROCESSING TIME VERIFICATION")
    print("-" * 40)
    completed_requests = df[df['status'] == 'completed'].copy()
    if len(completed_requests) > 0:
        completed_requests['processing_days'] = (completed_requests['finish_time'] - completed_requests['created_at']).dt.days
        print(f"Completed requests: {len(completed_requests):,}")
        print(f"Average processing time: {completed_requests['processing_days'].mean():.1f} days")
        print(f"Median processing time: {completed_requests['processing_days'].median():.1f} days")
        print(f"Min processing time: {completed_requests['processing_days'].min()} days")
        print(f"Max processing time: {completed_requests['processing_days'].max()} days")

    # Sample record verification
    print("\n🔍 SAMPLE RECORD VERIFICATION")
    print("-" * 40)
    sample_record = df.iloc[0]
    print("Sample record details:")
    for col in df.columns:
        value = sample_record[col]
        if isinstance(value, str) and len(value) > 50:
            print(f"  {col}: {value[:50]}...")
        else:
            print(f"  {col}: {value}")

    # JSON nodes verification
    print("\n📋 JSON NODES VERIFICATION")
    print("-" * 40)
    try:
        sample_nodes = df['nodes'].dropna().iloc[0]
        parsed_nodes = json.loads(sample_nodes)
        print(f"Sample nodes structure: {len(parsed_nodes)} nodes")
        if parsed_nodes:
            print(f"First node: {parsed_nodes[0]}")
    except Exception as e:
        print(f"JSON parsing error: {e}")

    # Data quality checks
    print("\n✅ DATA QUALITY CHECKS")
    print("-" * 40)
    print(f"Total records: {len(df):,}")
    print(f"Missing values per column:")
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            print(f"  {col}: {missing:,} missing ({missing/len(df)*100:.2f}%)")

    # Duplicate check
    print(f"\nDuplicate service_request_id: {df['service_request_id'].duplicated().sum()}")

    # Date consistency check
    print("\n📆 DATE CONSISTENCY CHECK")
    print("-" * 40)
    completed_with_dates = df[(df['status'] == 'completed') & df['finish_time'].notna()]
    inconsistent_dates = completed_with_dates[completed_with_dates['finish_time'] < completed_with_dates['created_at']]
    print(f"Requests finished before created: {len(inconsistent_dates)}")

    print("\n🎯 VERIFICATION COMPLETE")
    print("=" * 60)

    return df

if __name__ == "__main__":
    df_verified = verify_data_integrity()
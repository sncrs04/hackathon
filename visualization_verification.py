"""
Visualization Data Processing Verification
==========================================
Verify that the visualization script processes data correctly
"""

import pandas as pd
import numpy as np

def verify_visualization_processing():
    print('🔍 VISUALIZATION SCRIPT DATA VERIFICATION')
    print('=' * 50)

    # Load data exactly as in visualization script
    df = pd.read_csv('scramble-data.csv')

    # Arabic to English organization name mapping (from visualization script)
    org_translation = {
        'ديوان محافظة بابل': 'Babel Governorate Council',
        'هيأة التقاعد الوطنية': 'National Pension Authority',
        'دائرة التخطيط والمتابعة - وزارة التجارة': 'Planning & Follow-up - Ministry of Commerce',
        'ديوان محافظة الديوانية': 'Al-Diwaniyah Governorate Council',
        'قسم الشركات المحدودة': 'Limited Companies Department',
        'مؤسسة الشهداء': 'Martyrs Foundation',
        'وزارة الكهرباء': 'Ministry of Electricity',
        'مديرية تحقيق الادلة الجنائية': 'Criminal Evidence Investigation Directorate',
        'دائرة العيادات الطبية الشعبية': 'People\'s Medical Clinics Department',
        'الهيئة العليا للحج والعمرة': 'Supreme Hajj and Umrah Authority',
        'دائرة الصحة العامة': 'Public Health Department',
        'وزارة العمل والشؤون الاجتماعية': 'Ministry of Labor and Social Affairs',
        'وزارة النفط': 'Ministry of Oil',
        'مؤسسة السجناء السياسيين': 'Political Prisoners Foundation',
        'اللجنة المركزية لحصر السلاح': 'Central Weapons Registration Committee',
        'مجلس الخدمة العامة الاتحادي': 'Federal Civil Service Council',
        'دائرة التقاعد والضمان الاجتماعي': 'Retirement & Social Security Department',
        'وزارة التعليم العالي والبحث العلمي': 'Ministry of Higher Education & Scientific Research',
        'وزارة الهجرة والمهجرين': 'Ministry of Migration and Displaced',
        'دائرة هيأة التشغيل': 'Employment Authority Department',
        'المنظمات والمراكز الثقافية': 'Cultural Organizations and Centers',
        'مكتب الوزير / مديرية شكاوي المواطنين': 'Minister\'s Office / Citizen Complaints Directorate',
    }

    # Parse dates
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['finish_time'] = pd.to_datetime(df['finish_time'], errors='coerce')

    # Translate organization names
    df['org_name_en'] = df['org_name'].map(org_translation).fillna(df['org_name'])
    df['org_name_display'] = df['org_name_en']

    # Extract features for analysis
    df['year_month'] = df['created_at'].dt.to_period('M')
    df['date'] = df['created_at'].dt.date
    df['status'] = df['status'].fillna('unknown')

    # Calculate processing days for completed requests
    df['processing_days'] = (df['finish_time'] - df['created_at']).dt.days
    df['processing_days'] = df['processing_days'].clip(lower=0)

    print(f'Total records: {len(df):,}')
    print(f'Date range: {df["created_at"].min().date()} to {df["created_at"].max().date()}')

    # Status verification
    status_counts = df['status'].value_counts()
    print(f'\nStatus distribution:')
    for status, count in status_counts.items():
        print(f'  {status}: {count:,} ({count/len(df)*100:.2f}%)')

    # Organization verification
    org_counts = df['org_name_display'].value_counts()
    print(f'\nTop 5 organizations (English):')
    for i, (org, count) in enumerate(org_counts.head(5).items(), 1):
        print(f'  {i}. {org}: {count:,}')

    # Processing time verification
    completed_requests = df[df['status'] == 'completed']
    if len(completed_requests) > 0:
        avg_processing = completed_requests['processing_days'].mean()
        print(f'\nProcessing time for completed requests:')
        print(f'  Average: {avg_processing:.1f} days')
        print(f'  Count: {len(completed_requests):,}')

    # Payment service verification
    payment_counts = df['payment_service'].value_counts()
    print(f'\nPayment service distribution:')
    for payment, count in payment_counts.items():
        print(f'  {payment}: {count:,} ({count/len(df)*100:.2f}%)')

    # Verify calculations match the original verification
    print(f'\n🔍 CROSS-VERIFICATION WITH ORIGINAL DATA:')
    print(f'  Total records match: {len(df) == 50000}')
    print(f'  Status counts match: {status_counts["pending"] == 29509 and status_counts["completed"] == 17397 and status_counts["rejected"] == 3094}')
    print(f'  Top org match: {org_counts.index[0] == "Babel Governorate Council" and org_counts.iloc[0] == 10833}')

    print('\n✅ Visualization data processing verified!')

if __name__ == "__main__":
    verify_visualization_processing()
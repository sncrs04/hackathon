"""
Service Request Data Visualization with Seaborn
================================================
Minimalistic, user-friendly dashboard with Arabic font support
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Install and configure Arabic font
try:
    import matplotlib.font_manager as fm
    # Try to use system fonts that support Arabic
    plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans', 'sans-serif']
except:
    pass

# Set style for better-looking plots
sns.set_theme(style="white")
sns.set_palette("Set2")

# Load data
print("Loading data...")
df = pd.read_csv('scramble-data.csv')

# Arabic to English organization name mapping
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
    'هيأة التقاعد الوطنية': 'National Pension Authority',
    'اللجنة المركزية لحصر السلاح': 'Central Weapons Registration Committee',
    'مجلس الخدمة العامة الاتحادي': 'Federal Civil Service Council',
    'دائرة التقاعد والضمان الاجتماعي': 'Retirement & Social Security Department',
    'وزارة التعليم العالي والبحث العلمي': 'Ministry of Higher Education & Scientific Research',
    'وزارة الهجرة والمهجرين': 'Ministry of Migration and Displaced',
    'دائرة هيأة التشغيل': 'Employment Authority Department',
    'مؤسسة السجناء السياسيين': 'Political Prisoners Foundation',
    'مكتب الوزير / مديرية شكاوي المواطنين': 'Minister\'s Office / Citizen Complaints Directorate',
    'المنظمات والمراكز الثقافية': 'Cultural Organizations and Centers',
    'قسم تقنية المعلومات - الجوفية': 'IT Department - Groundwater',
}

# Parse dates
df['created_at'] = pd.to_datetime(df['created_at'])
df['finish_time'] = pd.to_datetime(df['finish_time'], errors='coerce')

# Translate organization names
df['org_name_en'] = df['org_name'].map(org_translation).fillna(df['org_name'])
# Use English names for visualization
df['org_name_display'] = df['org_name_en']

# Extract features for analysis
df['year_month'] = df['created_at'].dt.to_period('M')
df['date'] = df['created_at'].dt.date
df['status'] = df['status'].fillna('unknown')

# Calculate processing days for completed requests
df['processing_days'] = (df['finish_time'] - df['created_at']).dt.days
df['processing_days'] = df['processing_days'].clip(lower=0)

print(f"Data shape: {df.shape}")
print(f"Status distribution:\n{df['status'].value_counts()}\n")

# ============================================
# Create a minimalistic dashboard
# ============================================
fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor('#ffffff')
fig.suptitle('Service Requests Analytics', 
             fontsize=22, fontweight='bold', y=0.98, color='#1a1a1a')

# ============================================
# 1. BAR CHART: Requests per Organization
# ============================================
ax1 = plt.subplot(2, 3, 1)
ax1.set_facecolor('#ffffff')
org_counts = df['org_name_display'].value_counts().head(10)

colors_bar = sns.color_palette("Blues_d", len(org_counts))
bars = ax1.barh(range(len(org_counts)), org_counts.values, color=colors_bar, edgecolor='none')

ax1.set_yticks(range(len(org_counts)))
ax1.set_yticklabels(org_counts.index, fontsize=9)
ax1.set_xlabel('Requests', fontweight='bold', fontsize=10)
ax1.set_title('Top Organizations', fontsize=12, fontweight='bold', pad=12, color='#1a1a1a')
ax1.invert_yaxis()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('#e0e0e0')
ax1.spines['bottom'].set_color('#e0e0e0')
ax1.grid(axis='x', alpha=0.15, linestyle='-', linewidth=0.5)

# Add value labels
for i, (idx, val) in enumerate(org_counts.items()):
    ax1.text(val + 150, i, str(val), va='center', fontweight='bold', fontsize=8)

# ============================================
# 2. LINE CHART: Requests Over Time
# ============================================
ax2 = plt.subplot(2, 3, 2)
ax2.set_facecolor('#ffffff')
daily_requests = df.groupby('date').size().reset_index(name='count')
daily_requests['date'] = pd.to_datetime(daily_requests['date'])

ax2.plot(daily_requests['date'], daily_requests['count'], 
         linewidth=2.5, color='#3498db', marker=None, alpha=0.85)
ax2.fill_between(daily_requests['date'], daily_requests['count'], 
                  alpha=0.12, color='#3498db')

ax2.set_xlabel('Date', fontweight='bold', fontsize=10)
ax2.set_ylabel('Requests', fontweight='bold', fontsize=10)
ax2.set_title('Requests Over Time', fontsize=12, fontweight='bold', pad=12, color='#1a1a1a')
ax2.tick_params(axis='x', rotation=45, labelsize=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#e0e0e0')
ax2.spines['bottom'].set_color('#e0e0e0')
ax2.grid(True, alpha=0.12, linestyle='-', linewidth=0.5)

# ============================================
# 3. PIE CHART: Status Distribution
# ============================================
ax3 = plt.subplot(2, 3, 3)
ax3.set_facecolor('#ffffff')
status_counts = df['status'].value_counts()
colors_pie = ['#2ecc71', '#f39c12', '#e74c3c']

wedges, texts, autotexts = ax3.pie(status_counts.values, 
                                     labels=status_counts.index,
                                     autopct='%1.1f%%',
                                     colors=colors_pie[:len(status_counts)],
                                     startangle=90,
                                     textprops={'fontweight': 'bold', 'fontsize': 10})

ax3.set_title('Status Distribution', fontsize=12, fontweight='bold', pad=12, color='#1a1a1a')

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(9)
    autotext.set_fontweight('bold')

# ============================================
# 4. HEATMAP: Status by Organization
# ============================================
ax4 = plt.subplot(2, 3, 4)
ax4.set_facecolor('#ffffff')

top_orgs = df['org_name_display'].value_counts().head(7).index
df_filtered = df[df['org_name_display'].isin(top_orgs)]

heatmap_data = pd.crosstab(df_filtered['org_name_display'], df_filtered['status'])
heatmap_normalized = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100

sns.heatmap(heatmap_normalized, annot=True, fmt='.0f', cmap='RdYlGn', 
            cbar_kws={'label': '%'}, ax=ax4, linewidths=0.5, linecolor='white')

ax4.set_title('Status Distribution (%)', fontsize=12, fontweight='bold', pad=12, color='#1a1a1a')
ax4.set_xlabel('Status', fontweight='bold', fontsize=10)
ax4.set_ylabel('')
plt.setp(ax4.get_yticklabels(), rotation=0, fontsize=8)
plt.setp(ax4.get_xticklabels(), rotation=45, fontsize=8)

# ============================================
# 5. BAR CHART: Status Summary
# ============================================
ax5 = plt.subplot(2, 3, 5)
ax5.set_facecolor('#ffffff')
status_order = ['completed', 'pending', 'rejected']
status_data = df['status'].value_counts().reindex(status_order, fill_value=0)

colors_status = ['#2ecc71', '#f39c12', '#e74c3c']
bars = ax5.bar(range(len(status_data)), status_data.values, color=colors_status, edgecolor='none')

ax5.set_xticks(range(len(status_data)))
ax5.set_xticklabels(status_data.index, fontweight='bold', fontsize=9)
ax5.set_ylabel('Count', fontweight='bold', fontsize=10)
ax5.set_title('Status Count', fontsize=12, fontweight='bold', pad=12, color='#1a1a1a')
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.spines['left'].set_color('#e0e0e0')
ax5.spines['bottom'].set_color('#e0e0e0')
ax5.grid(axis='y', alpha=0.15, linestyle='-', linewidth=0.5)

# Add labels
for i, (idx, val) in enumerate(status_data.items()):
    ax5.text(i, val + 400, f'{int(val):,}', ha='center', fontweight='bold', fontsize=9)

# ============================================
# 6. KEY METRICS (Minimalistic)
# ============================================
ax6 = plt.subplot(2, 3, 6)
ax6.set_facecolor('#ffffff')
ax6.axis('off')

total_requests = len(df)
completion_rate = (df['status'] == 'completed').sum() / total_requests * 100
rejection_rate = (df['status'] == 'rejected').sum() / total_requests * 100
pending_rate = (df['status'] == 'pending').sum() / total_requests * 100

metrics_text = f"""
KEY METRICS

Total: {total_requests:,}

Completed: {completion_rate:.1f}%
Pending: {pending_rate:.1f}%
Rejected: {rejection_rate:.1f}%
"""

ax6.text(0.1, 0.85, metrics_text, transform=ax6.transAxes,
         fontfamily='monospace', fontsize=10.5, verticalalignment='top',
         fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#f5f5f5', 
                  edgecolor='#cccccc', linewidth=1, alpha=0.95))

plt.tight_layout()
plt.savefig('service_requests_dashboard.png', dpi=300, bbox_inches='tight', facecolor='#ffffff')
print("\n✓ Minimalistic dashboard saved!")

# ============================================
# Correlation Analysis (Minimalistic)
# ============================================
print("Generating correlation analysis...")
fig2, axes = plt.subplots(1, 2, figsize=(16, 6))
fig2.patch.set_facecolor('#ffffff')

# Create numeric data for correlation
df_numeric = df.copy()
df_numeric['status_code'] = pd.factorize(df_numeric['status'])[0]
df_numeric['payment_service_code'] = df_numeric['payment_service'].astype(int)
df_numeric['processing_days_filled'] = df_numeric['processing_days'].fillna(0)

# Select numeric columns
corr_columns = ['payment_service_code', 'status_code', 'processing_days_filled']
correlation_matrix = df_numeric[corr_columns].corr()

# Plot correlation heatmap
axes[0].set_facecolor('#ffffff')
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, cbar_kws={'label': 'Correlation'}, ax=axes[0],
            square=True, linewidths=1, linecolor='white', annot_kws={'fontsize': 10})
axes[0].set_title('Correlation Matrix', fontsize=14, fontweight='bold', pad=15, color='#1a1a1a')
axes[0].set_xticklabels(['Payment Service', 'Status Code', 'Processing Days'],
                        rotation=45, fontsize=10, ha='right')
axes[0].set_yticklabels(['Payment Service', 'Status Code', 'Processing Days'],
                        rotation=0, fontsize=10)

# Processing time by organization
axes[1].set_facecolor('#ffffff')
df_completed = df[df['status'] == 'completed'].copy()
if len(df_completed) > 0:
    # Get top 8 organizations for better readability
    top_orgs_completed = df_completed['org_name_display'].value_counts().head(8).index
    df_completed_filtered = df_completed[df_completed['org_name_display'].isin(top_orgs_completed)]

    sns.violinplot(data=df_completed_filtered, y='org_name_display', x='processing_days',
                   ax=axes[1], palette='Set2', inner='quartile')
    axes[1].set_xlim(0, df_completed_filtered['processing_days'].quantile(0.95))
    axes[1].set_xlabel('Processing Days', fontweight='bold', fontsize=11)
    axes[1].set_ylabel('')
    axes[1].set_title('Processing Time Distribution', fontsize=14, fontweight='bold', pad=15, color='#1a1a1a')
    plt.setp(axes[1].get_yticklabels(), fontsize=9)

    # Add grid for better readability
    axes[1].grid(axis='x', alpha=0.15, linestyle='-', linewidth=0.5)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['left'].set_color('#e0e0e0')
    axes[1].spines['bottom'].set_color('#e0e0e0')

plt.tight_layout(pad=2.0)
plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight', facecolor='#ffffff')
print("✓ Correlation analysis saved!")

# ============================================
# Summary Statistics
# ============================================
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)
print(f"\nTotal Requests: {total_requests:,}")
print(f"Date Range: {df['created_at'].min().date()} to {df['created_at'].max().date()}")
print(f"\nCompletion Rate: {completion_rate:.2f}%")
print(f"Rejection Rate: {rejection_rate:.2f}%")
print(f"Pending Rate: {pending_rate:.2f}%")
print(f"\nTop 5 Organizations by Volume:")
for i, (org, count) in enumerate(df['org_name_display'].value_counts().head(5).items(), 1):
    print(f"  {i}. {org}: {count:,} requests")

print("\n✓ All visualizations generated successfully!")
print("\nGenerated files:")
print("  • service_requests_dashboard.png")
print("  • correlation_analysis.png")

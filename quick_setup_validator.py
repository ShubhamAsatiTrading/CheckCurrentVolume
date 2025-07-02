# quick_setup_validator.py - Pre-integration environment validation
# Run this before integrating new dashboard features

import os
import sys
from datetime import datetime

def check_file_structure():
    """Check if all required files are present"""
    print("📁 Checking File Structure...")
    
    required_files = {
        'main.py': 'Main dashboard file',
        'volume_average.py': 'Volume average calculation module',
        'common.txt': 'Configuration file',
        'symbols.txt': 'Stock symbols list'
    }
    
    optional_files = {
        'live_data_downloader_parallel.py': 'Parallel live data downloader',
        'live_data_downloader.py': 'Original live data downloader',
        '.env': 'Environment variables',
        'kite_token.txt': 'Kite authentication token'
    }
    
    success_count = 0
    
    for file, description in required_files.items():
        if os.path.exists(file):
            print(f"✅ {file:<25} - {description}")
            success_count += 1
        else:
            print(f"❌ {file:<25} - {description} (REQUIRED)")
    
    print("\nOptional Files:")
    for file, description in optional_files.items():
        if os.path.exists(file):
            print(f"✅ {file:<25} - {description}")
        else:
            print(f"⚪ {file:<25} - {description} (Optional)")
    
    return success_count == len(required_files)

def check_dependencies():
    """Check if required Python packages are installed"""
    print("\n📦 Checking Dependencies...")
    
    required_packages = {
        'streamlit': 'Web dashboard framework',
        'pandas': 'Data manipulation library',
        'datetime': 'Date/time handling (built-in)',
        'json': 'JSON handling (built-in)',
        'os': 'Operating system interface (built-in)',
        'glob': 'File pattern matching (built-in)'
    }
    
    optional_packages = {
        'openpyxl': 'Excel file support',
        'kiteconnect': 'Kite API client',
        'dotenv': 'Environment variables',
        'aiohttp': 'Async HTTP client (for parallel downloader)'
    }
    
    success_count = 0
    total_required = len([pkg for pkg in required_packages.keys() if pkg not in ['datetime', 'json', 'os', 'glob']])
    
    for package, description in required_packages.items():
        try:
            if package == 'datetime':
                import datetime
            elif package == 'json':
                import json
            elif package == 'os':
                import os
            elif package == 'glob':
                import glob
            elif package == 'streamlit':
                import streamlit
            elif package == 'pandas':
                import pandas
            
            print(f"✅ {package:<15} - {description}")
            if package not in ['datetime', 'json', 'os', 'glob']:
                success_count += 1
                
        except ImportError:
            print(f"❌ {package:<15} - {description} (INSTALL REQUIRED)")
    
    print("\nOptional Packages:")
    for package, description in optional_packages.items():
        try:
            if package == 'openpyxl':
                import openpyxl
            elif package == 'kiteconnect':
                import kiteconnect
            elif package == 'dotenv':
                import dotenv
            elif package == 'aiohttp':
                import aiohttp
            
            print(f"✅ {package:<15} - {description}")
        except ImportError:
            print(f"⚪ {package:<15} - {description} (Optional)")
    
    return success_count >= total_required

def check_configuration():
    """Check configuration file content"""
    print("\n⚙️ Checking Configuration...")
    
    if not os.path.exists('common.txt'):
        print("❌ common.txt not found")
        return False
    
    # Read current configuration
    config = {}
    try:
        with open('common.txt', 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    config[key] = value
    except Exception as e:
        print(f"❌ Error reading common.txt: {e}")
        return False
    
    # Check required parameters
    required_params = {
        'stop_loss': 'Stop loss percentage',
        'target': 'Target profit percentage',
        'ohlc_value': 'OHLC value to use'
    }
    
    new_params = {
        'avg_volume_days': 'Volume average calculation days',
        'live_data_download': 'Live data download enable/disable',
        'rerun_minute': 'Live data update interval'
    }
    
    # Check existing parameters
    missing_required = []
    for param, description in required_params.items():
        if param in config:
            print(f"✅ {param:<20} = {config[param]:<10} - {description}")
        else:
            print(f"❌ {param:<20} = {'MISSING':<10} - {description}")
            missing_required.append(param)
    
    # Check new parameters
    missing_new = []
    print("\nNew Parameters (will be added):")
    for param, description in new_params.items():
        if param in config:
            print(f"✅ {param:<20} = {config[param]:<10} - {description}")
        else:
            print(f"➕ {param:<20} = {'TO ADD':<10} - {description}")
            missing_new.append(param)
    
    if missing_new:
        print(f"\n💡 Will add {len(missing_new)} new parameters during integration")
    
    return len(missing_required) == 0

def check_symbols_file():
    """Check symbols file content"""
    print("\n📊 Checking Symbols File...")
    
    if not os.path.exists('symbols.txt'):
        print("❌ symbols.txt not found")
        return False
    
    try:
        with open('symbols.txt', 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
        
        if symbols:
            print(f"✅ Found {len(symbols)} symbols")
            print(f"📋 Sample symbols: {', '.join(symbols[:5])}" + ("..." if len(symbols) > 5 else ""))
            
            if len(symbols) > 50:
                print("⚠️  Large symbol count detected - consider testing with smaller set first")
            elif len(symbols) < 5:
                print("⚠️  Small symbol count - consider adding more symbols for better testing")
            
            return True
        else:
            print("❌ symbols.txt is empty")
            return False
            
    except Exception as e:
        print(f"❌ Error reading symbols.txt: {e}")
        return False

def check_existing_data():
    """Check for existing data files"""
    print("\n📂 Checking Existing Data...")
    
    # Check for historical data
    if os.path.exists('stocks_historical_data'):
        hist_files = [f for f in os.listdir('stocks_historical_data') if f.endswith('_historical.csv')]
        if hist_files:
            print(f"✅ Historical data: {len(hist_files)} files found")
        else:
            print("⚪ Historical data: No files found (will need to download)")
    else:
        print("⚪ Historical data folder: Not found (will be created)")
    
    # Check for average data
    if os.path.exists('Average data'):
        avg_files = [f for f in os.listdir('Average data') if f.startswith('AvgData_')]
        if avg_files:
            latest_avg = max(avg_files, key=lambda x: os.path.getctime(os.path.join('Average data', x)))
            print(f"✅ Average data: Found {len(avg_files)} files, latest: {latest_avg}")
        else:
            print("⚪ Average data: Folder exists but no data files")
    else:
        print("⚪ Average data folder: Not found (will be created)")
    
    # Check for live data
    today = datetime.now().strftime("%Y-%m-%d")
    today_file = f"StockLiveData_{today}.csv"
    
    live_files = [f for f in os.listdir('.') if f.startswith('StockLiveData_')]
    if live_files:
        print(f"✅ Live data: Found {len(live_files)} files")
        if today_file in live_files:
            print(f"✅ Today's live data: {today_file} exists")
        else:
            print(f"⚪ Today's live data: {today_file} not found")
    else:
        print("⚪ Live data: No files found (run live downloader to generate)")

def suggest_integration_steps():
    """Provide next steps for integration"""
    print("\n🚀 Integration Steps...")
    
    print("1. ✅ Add imports to main.py:")
    print("   from volume_average import VolumeAverage")
    print("   import subprocess, sys")
    
    print("\n2. ✅ Add Volume Average to Individual Function Controls:")
    print("   Copy the 'Volume Average Analysis' expander")
    
    print("\n3. ✅ Add Live Data Control to Individual Function Controls:")
    print("   Copy the 'Live Data Downloader Control' expander")
    
    print("\n4. ✅ Update session state initialization:")
    print("   Add 'volume_change_data', 'volume_change_collapsed', 'trading_system_collapsed'")
    
    print("\n5. ✅ Make Complete Trading System collapsible:")
    print("   Add expand/collapse button to section header")
    
    print("\n6. ✅ Add Live Volume Change section:")
    print("   Copy complete section after Complete Trading System")
    
    print("\n7. ✅ Test in Streamlit:")
    print("   streamlit run main.py")

def create_missing_config():
    """Create missing configuration parameters"""
    print("\n🔧 Updating Configuration...")
    
    if not os.path.exists('common.txt'):
        print("Creating new common.txt...")
        with open('common.txt', 'w') as f:
            f.write("# Trading System Configuration\n")
            f.write("stop_loss=4.0\n")
            f.write("target=10.0\n")
            f.write("ohlc_value=open\n")
            f.write("trade_today_flag=no\n")
            f.write("avg_volume_days=30\n")
            f.write("live_data_download=no\n")
            f.write("rerun_minute=1\n")
            f.write("symbols_per_worker=5\n")
        print("✅ Created common.txt with default values")
        return
    
    # Read existing config
    with open('common.txt', 'r') as f:
        lines = f.readlines()
    
    # Check what's missing
    content = ''.join(lines)
    new_params = {
        'avg_volume_days': '30',
        'live_data_download': 'no', 
        'rerun_minute': '1',
        'symbols_per_worker': '5'
    }
    
    # Add missing parameters
    added = []
    for param, default_value in new_params.items():
        if f"{param}=" not in content:
            lines.append(f"{param}={default_value}\n")
            added.append(param)
    
    if added:
        # Write back updated config
        with open('common.txt', 'w') as f:
            f.writelines(lines)
        print(f"✅ Added {len(added)} new parameters: {', '.join(added)}")
    else:
        print("✅ All required parameters already present")

def main():
    """Main validation function"""
    print("🔍 PRE-INTEGRATION ENVIRONMENT VALIDATION")
    print("=" * 50)
    print(f"⏰ Validation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Run all checks
    checks = {
        'File Structure': check_file_structure(),
        'Dependencies': check_dependencies(),
        'Configuration': check_configuration(),
        'Symbols File': check_symbols_file()
    }
    
    # Additional checks (informational)
    check_existing_data()
    
    # Results summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION RESULTS")
    print("=" * 50)
    
    passed = 0
    total = len(checks)
    
    for check_name, result in checks.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{check_name:<20} {status}")
        if result:
            passed += 1
    
    print("-" * 50)
    print(f"OVERALL: {passed}/{total} checks passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ENVIRONMENT READY FOR INTEGRATION!")
        create_missing_config()
        suggest_integration_steps()
        
        print("\n💡 Next Steps:")
        print("1. Follow the integration steps above")
        print("2. Run: python test_dashboard_integration.py")
        print("3. Test manually in Streamlit dashboard")
        
    elif passed >= total - 1:
        print("\n⚠️ MOSTLY READY - Fix minor issues above")
        print("Integration should work with minor adjustments")
        
    else:
        print("\n❌ NOT READY - Fix critical issues before integration")
        print("Address the failed checks above before proceeding")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
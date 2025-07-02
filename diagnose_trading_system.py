# diagnose_trading_system.py - Compare direct vs web interface execution
# Run this to identify differences: python diagnose_trading_system.py

import os
import sys
import subprocess
import json
from datetime import datetime
from dotenv import load_dotenv

def get_environment_snapshot():
    """Get complete environment snapshot"""
    load_dotenv()
    
    return {
        "python_info": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": sys.platform,
            "path": sys.path[:5]  # First 5 paths
        },
        "working_directory": os.getcwd(),
        "environment_variables": {
            "KITE_API_KEY": "***SET***" if os.getenv('KITE_API_KEY') else "NOT SET",
            "KITE_API_SECRET": "***SET***" if os.getenv('KITE_API_SECRET') else "NOT SET",
            "TELEGRAM_BOT_TOKEN": "***SET***" if os.getenv('TELEGRAM_BOT_TOKEN') else "NOT SET",
            "TELEGRAM_CHAT_ID": "***SET***" if os.getenv('TELEGRAM_CHAT_ID') else "NOT SET",
            "PYTHONPATH": os.getenv('PYTHONPATH', 'NOT SET'),
            "PATH": os.getenv('PATH', 'NOT SET')[:200] + "..." if os.getenv('PATH') else "NOT SET"
        },
        "files_status": {
            "code_1.py": {
                "exists": os.path.exists('code_1.py'),
                "size": os.path.getsize('code_1.py') if os.path.exists('code_1.py') else 0,
                "modified": datetime.fromtimestamp(os.path.getmtime('code_1.py')).isoformat() if os.path.exists('code_1.py') else None
            },
            "main.py": {
                "exists": os.path.exists('main.py'),
                "size": os.path.getsize('main.py') if os.path.exists('main.py') else 0
            },
            ".env": {
                "exists": os.path.exists('.env'),
                "size": os.path.getsize('.env') if os.path.exists('.env') else 0
            },
            "common.txt": {
                "exists": os.path.exists('common.txt'),
                "content": open('common.txt', 'r').read() if os.path.exists('common.txt') else None
            },
            "symbols.txt": {
                "exists": os.path.exists('symbols.txt'),
                "count": len([line.strip() for line in open('symbols.txt', 'r') if line.strip()]) if os.path.exists('symbols.txt') else 0
            },
            "kite_token.txt": {
                "exists": os.path.exists('kite_token.txt'),
                "valid": check_token_validity() if os.path.exists('kite_token.txt') else False
            }
        },
        "directories": {
            name: os.path.exists(name) for name in [
                "stocks_historical_data", "aggregated_data", 
                "Volume_boost_consolidated", "backtest_results"
            ]
        },
        "python_packages": check_python_packages(),
        "permissions": {
            "current_dir_writable": os.access('.', os.W_OK),
            "code_1_readable": os.access('code_1.py', os.R_OK) if os.path.exists('code_1.py') else False,
            "code_1_executable": os.access('code_1.py', os.X_OK) if os.path.exists('code_1.py') else False
        }
    }

def check_token_validity():
    """Check if saved token is valid"""
    try:
        with open('kite_token.txt', 'r') as f:
            token_data = json.loads(f.read().strip())
            today = datetime.now().strftime("%Y-%m-%d")
            return token_data.get("date") == today and bool(token_data.get("access_token"))
    except:
        return False

def check_python_packages():
    """Check Python packages status"""
    packages = {}
    
    package_list = [
        'streamlit', 'pandas', 'dotenv', 'kiteconnect', 
        'requests', 'numpy', 'matplotlib'
    ]
    
    for package in package_list:
        try:
            if package == 'dotenv':
                import_name = 'dotenv'
                module = __import__('dotenv')
            else:
                module = __import__(package)
            
            version = getattr(module, '__version__', 'Unknown')
            packages[package] = {"installed": True, "version": version}
        except ImportError:
            packages[package] = {"installed": False, "version": None}
    
    return packages

def test_direct_execution():
    """Test direct execution of code_1.py"""
    print("[TEST] Testing direct execution of code_1.py...")
    
    if not os.path.exists('code_1.py'):
        return {
            "success": False,
            "error": "code_1.py not found",
            "output": "",
            "duration": 0
        }
    
    start_time = datetime.now()
    
    try:
        # Run with timeout to prevent hanging
        result = subprocess.run(
            [sys.executable, 'code_1.py'],
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout for quick test
            cwd=os.getcwd()
        )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return {
            "success": result.returncode == 0,
            "return_code": result.returncode,
            "output": result.stdout[:1000] + "..." if len(result.stdout) > 1000 else result.stdout,
            "error": result.stderr[:1000] + "..." if len(result.stderr) > 1000 else result.stderr,
            "duration": duration
        }
        
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Process timed out after 30 seconds (this might be normal)",
            "output": "",
            "duration": 30
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "output": "",
            "duration": (datetime.now() - start_time).total_seconds()
        }

def test_web_interface_environment():
    """Test what environment the web interface would see"""
    print("[WEB] Testing web interface environment simulation...")
    
    # Simulate Streamlit environment
    env = os.environ.copy()
    
    # Add typical Streamlit modifications
    env['STREAMLIT_SERVER_PORT'] = '8501'
    env['STREAMLIT_SERVER_HEADLESS'] = 'true'
    
    # Test subprocess call like web interface does
    try:
        result = subprocess.run(
            [sys.executable, '-c', '''
import os, sys
print("Python executable:", sys.executable)
print("Working directory:", os.getcwd())
print("Python path:", sys.path[:3])
print("KITE_API_KEY set:", bool(os.getenv("KITE_API_KEY")))
print("code_1.py exists:", os.path.exists("code_1.py"))

# Try importing key modules
try:
    import kiteconnect
    print("kiteconnect import: SUCCESS")
except Exception as e:
    print("kiteconnect import: FAILED -", e)

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("dotenv load: SUCCESS")
except Exception as e:
    print("dotenv load: FAILED -", e)
            '''],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
            cwd=os.getcwd()
        )
        
        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr
        }
        
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e)
        }

def create_diagnostic_report():
    """Create comprehensive diagnostic report"""
    print("[SEARCH] TRADING SYSTEM DIAGNOSTIC REPORT")
    print("=" * 60)
    
    # Get environment snapshot
    print("[DATA] Gathering environment information...")
    env_snapshot = get_environment_snapshot()
    
    # Test direct execution
    print("[TEST] Testing direct execution...")
    direct_test = test_direct_execution()
    
    # Test web interface environment
    print("[WEB] Testing web interface environment...")
    web_test = test_web_interface_environment()
    
    # Create report
    report = {
        "timestamp": datetime.now().isoformat(),
        "environment": env_snapshot,
        "direct_execution_test": direct_test,
        "web_interface_environment_test": web_test,
        "analysis": analyze_differences(env_snapshot, direct_test, web_test)
    }
    
    # Save report to file
    report_file = f"diagnostic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"[LOG] Report saved to: {os.path.abspath(report_file)}")
    except Exception as e:
        print(f"[ERROR] Failed to save report: {e}")
    
    # Display summary
    display_summary(report)
    
    return report

def analyze_differences(env_snapshot, direct_test, web_test):
    """Analyze differences and provide recommendations"""
    issues = []
    recommendations = []
    
    # Check if direct execution works
    if not direct_test["success"] and direct_test.get("return_code") != 0:
        issues.append("Direct execution of code_1.py failed")
        recommendations.append("Fix code_1.py issues first before testing web interface")
    
    # Check Python packages
    packages = env_snapshot["python_packages"]
    missing_packages = [name for name, info in packages.items() if not info["installed"]]
    
    if missing_packages:
        issues.append(f"Missing Python packages: {', '.join(missing_packages)}")
        recommendations.append(f"Install missing packages: pip install {' '.join(missing_packages)}")
    
    # Check environment variables
    env_vars = env_snapshot["environment_variables"]
    missing_env = [name for name, value in env_vars.items() if value == "NOT SET" and name in ["KITE_API_KEY", "KITE_API_SECRET"]]
    
    if missing_env:
        issues.append(f"Missing environment variables: {', '.join(missing_env)}")
        recommendations.append("Update .env file with correct Kite Connect credentials")
    
    # Check files
    files = env_snapshot["files_status"]
    if not files["code_1.py"]["exists"]:
        issues.append("code_1.py file not found")
        recommendations.append("Ensure code_1.py is in the current directory")
    
    if not files["kite_token.txt"]["valid"]:
        issues.append("No valid Kite access token")
        recommendations.append("Authenticate via web interface to get valid token")
    
    # Check web interface environment
    if not web_test["success"]:
        issues.append("Web interface environment test failed")
        recommendations.append("Check if Streamlit is properly installed and configured")
    
    # Check permissions
    permissions = env_snapshot["permissions"]
    if not permissions["current_dir_writable"]:
        issues.append("Current directory is not writable")
        recommendations.append("Ensure you have write permissions in the current directory")
    
    return {
        "issues_found": issues,
        "recommendations": recommendations,
        "severity": "HIGH" if len(issues) > 3 else "MEDIUM" if len(issues) > 1 else "LOW"
    }

def display_summary(report):
    """Display diagnostic summary"""
    print("\n" + "=" * 60)
    print("[INFO] DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    # Environment status
    env = report["environment"]
    print(f"[PYTHON] Python: {env['python_info']['version'].split()[0]}")
    print(f" Working Directory: {env['working_directory']}")
    
    # Package status
    packages = env["python_packages"]
    installed = sum(1 for p in packages.values() if p["installed"])
    total = len(packages)
    print(f"[PACKAGE] Packages: {installed}/{total} installed")
    
    # Files status
    files = env["files_status"]
    print(f"[FILE] Key Files:")
    for name, info in files.items():
        status = "[SUCCESS]" if info["exists"] else "[ERROR]"
        print(f"   {status} {name}")
    
    # Test results
    direct = report["direct_execution_test"]
    web = report["web_interface_environment_test"]
    
    print(f"\n[TEST] Direct Execution: {'[SUCCESS] SUCCESS' if direct['success'] else '[ERROR] FAILED'}")
    if not direct["success"]:
        print(f"   Error: {direct.get('error', 'Unknown error')}")
    
    print(f"[WEB] Web Environment: {'[SUCCESS] SUCCESS' if web['success'] else '[ERROR] FAILED'}")
    if not web["success"]:
        print(f"   Error: {web.get('error', 'Unknown error')}")
    
    # Analysis
    analysis = report["analysis"]
    print(f"\n[SEARCH] Issues Found: {len(analysis['issues_found'])} (Severity: {analysis['severity']})")
    
    if analysis["issues_found"]:
        print("\n[ERROR] ISSUES:")
        for i, issue in enumerate(analysis["issues_found"], 1):
            print(f"   {i}. {issue}")
        
        print("\n[TIP] RECOMMENDATIONS:")
        for i, rec in enumerate(analysis["recommendations"], 1):
            print(f"   {i}. {rec}")
    else:
        print("[SUCCESS] No major issues detected!")
    
    print("\n" + "=" * 60)

def main():
    """Main diagnostic function"""
    try:
        report = create_diagnostic_report()
        
        print("\n[TIP] NEXT STEPS:")
        
        analysis = report["analysis"]
        if analysis["issues_found"]:
            print("1. Fix the issues listed above")
            print("2. Re-run this diagnostic script")
            print("3. Test both direct execution and web interface")
        else:
            print("1. Try running the web interface: python run.py")
            print("2. If still having issues, check the detailed report file")
        
        print(f"\n[LOG] Detailed report saved for further analysis")
        
    except Exception as e:
        print(f"[CRITICAL] Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()


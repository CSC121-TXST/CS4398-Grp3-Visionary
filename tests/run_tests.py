#!/usr/bin/env python3
"""
Test Runner for Visionary Project

Runs all tests and generates comprehensive reports.
"""

import sys
import subprocess
import json
import re
from pathlib import Path
from datetime import datetime


def run_tests():
    """Run all tests and generate reports."""
    project_root = Path(__file__).parent.parent
    
    print("=" * 80)
    print("Visionary Test Suite")
    print("=" * 80)
    print(f"Running tests at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Test results storage
    results = {
        "timestamp": datetime.now().isoformat(),
        "tests_run": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "tests_skipped": 0,
        "coverage": None,
        "test_details": []
    }
    
    # Create reports directory first
    reports_dir = project_root / "tests" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        f"--junitxml={reports_dir / 'junit.xml'}",
        "--cov=src",
        f"--cov-report=html:{reports_dir / 'coverage'}",
        "--cov-report=term",
        f"--cov-report=json:{reports_dir / 'coverage.json'}"
    ]
    
    # Add HTML report if pytest-html is available
    try:
        import pytest_html
        cmd.extend([
            f"--html={reports_dir / 'report.html'}",
            "--self-contained-html"
        ])
        html_available = True
    except ImportError:
        html_available = False
    
    try:
        print("Running unit and integration tests...")
        print("-" * 80)
        print()
        
        # Run pytest with real-time output
        result = subprocess.run(cmd, cwd=project_root)
        
        # Parse results from JUnit XML (most reliable)
        junit_file = reports_dir / 'junit.xml'
        if junit_file.exists():
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(junit_file)
                root = tree.getroot()
                
                # Handle both testsuites and testsuite root elements
                if root.tag == 'testsuites':
                    results["tests_run"] = int(root.get('tests', 0))
                    results["tests_passed"] = int(root.get('tests', 0)) - int(root.get('failures', 0)) - int(root.get('errors', 0))
                    results["tests_failed"] = int(root.get('failures', 0)) + int(root.get('errors', 0))
                    results["tests_skipped"] = int(root.get('skipped', 0))
                elif root.tag == 'testsuite':
                    results["tests_run"] = int(root.get('tests', 0))
                    results["tests_passed"] = int(root.get('tests', 0)) - int(root.get('failures', 0)) - int(root.get('errors', 0))
                    results["tests_failed"] = int(root.get('failures', 0)) + int(root.get('errors', 0))
                    results["tests_skipped"] = int(root.get('skipped', 0))
            except Exception as e:
                print(f"Warning: Could not parse JUnit XML: {e}")
        
        # Also try parsing from stdout as backup if JUnit XML didn't work
        if results["tests_run"] == 0:
            # Re-run with capture to parse
            parse_cmd = [c for c in cmd if not c.startswith('--html') and c != '--self-contained-html']
            parse_result = subprocess.run(parse_cmd, capture_output=True, text=True, cwd=project_root)
            output_text = parse_result.stdout + parse_result.stderr
            
            # Parse output for test counts as backup
            for line in output_text.split('\n'):
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in ['passed', 'failed', 'error', 'skipped']):
                    # Extract passed count
                    match = re.search(r'(\d+)\s+passed', line_lower)
                    if match:
                        results["tests_passed"] = int(match.group(1))
                    
                    # Extract failed count
                    match = re.search(r'(\d+)\s+failed', line_lower)
                    if match:
                        results["tests_failed"] = int(match.group(1))
                    
                    # Extract error count
                    match = re.search(r'(\d+)\s+error', line_lower)
                    if match:
                        results["tests_failed"] += int(match.group(1))
                    
                    # Extract skipped count
                    match = re.search(r'(\d+)\s+skipped', line_lower)
                    if match:
                        results["tests_skipped"] = int(match.group(1))
            
            results["tests_run"] = results["tests_passed"] + results["tests_failed"] + results["tests_skipped"]
        
        # Load coverage data if available
        coverage_file = reports_dir / "coverage.json"
        if coverage_file.exists():
            try:
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                    results["coverage"] = coverage_data.get("totals", {}).get("percent_covered")
            except Exception:
                pass
        
        # Print summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"Tests Run:    {results['tests_run']}")
        print(f"Tests Passed: {results['tests_passed']}")
        print(f"Tests Failed: {results['tests_failed']}")
        if results["tests_skipped"] > 0:
            print(f"Tests Skipped: {results['tests_skipped']}")
        if results["coverage"]:
            print(f"Coverage:     {results['coverage']:.2f}%")
        print("=" * 80)
        
        # Save results to JSON
        results_file = reports_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nReports generated:")
        if html_available and (reports_dir / 'report.html').exists():
            print(f"  - HTML Report: {reports_dir / 'report.html'}")
        print(f"  - Coverage Report: {reports_dir / 'coverage' / 'index.html'}")
        print(f"  - JUnit XML: {reports_dir / 'junit.xml'}")
        print(f"  - Results JSON: {reports_dir / 'test_results.json'}")
        
        if not html_available:
            print(f"\nNote: Install pytest-html for HTML reports: pip install pytest-html")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running tests: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

import subprocess
import sys
import os
import re
import time
import csv
from datetime import datetime
from tqdm import tqdm

class AutomatedTester:
    def __init__(self, num_runs=5):
        self.num_runs = num_runs
        self.txt_log_file = r"G:\My Drive\TradingBot\Programming\Train Scripts (With Features)\logs\perf_metrics.txt"
        self.csv_log_file = r"G:\My Drive\TradingBot\Programming\Train Scripts (With Features)\logs\perf_metrics.csv"
        
    def extract_metrics(self, output):
        metrics = {}
        patterns = {
            'Total Return %': r'Total Return %:[\s]*([-\d.]+)',
            'Sharpe Ratio': r'Sharpe Ratio:[\s]*([-\d.]+)',
            'Max Drawdown %': r'(?:Max(?:imum)?[ _]?Drawdown(?: %|%)?:[\s]*([-\d.]+))',  # Matches Max Drawdown %: <value>
            'Last Train Loss': r'Epoch \d+/\d+ - Train:[\s]*([\d.]+)\s*\|\s*Val:[\s]*[\d.]+',
            'Last Val Loss': r'Epoch \d+/\d+ - Train:[\s]*[\d.]+.*\|\s*Val:[\s]*([\d.]+)'
        }
        
        for key, pattern in patterns.items():
            matches = re.findall(pattern, output, re.MULTILINE)
            if matches:
                metrics[key] = float(matches[-1])
            else:
                metrics[key] = None
                print(f"Warning: Could not extract {key} from output")
                
        return metrics
    
    def log_metrics(self, run_num, metrics, elapsed_time):
        """Log metrics to both TXT and CSV files"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # TXT logging
        with open(self.txt_log_file, 'a') as f:
            f.write(f"\n{'='*40}\n")
            f.write(f"Run #{run_num} | {timestamp}\n")
            f.write(f"Execution Time: {elapsed_time:.2f} seconds\n")
            f.write(f"-"*20 + "\n")
            
            for key, value in metrics.items():
                if value is not None:
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: N/A\n")
        
        # CSV logging
        headers = ['Run', 'Timestamp', 'Execution Time (s)', 'Total Return %', 'Sharpe Ratio', 'Max Drawdown %', 'Last Train Loss', 'Last Val Loss']
        row = {
            'Run': run_num,
            'Timestamp': timestamp,
            'Execution Time (s)': f"{elapsed_time:.2f}",
            'Total Return %': f"{metrics.get('Total Return %', 'N/A'):.4f}" if metrics.get('Total Return %') is not None else 'N/A',
            'Sharpe Ratio': f"{metrics.get('Sharpe Ratio', 'N/A'):.4f}" if metrics.get('Sharpe Ratio') is not None else 'N/A',
            'Max Drawdown %': f"{metrics.get('Max Drawdown %', 'N/A'):.4f}" if metrics.get('Max Drawdown %') is not None else 'N/A',
            'Last Train Loss': f"{metrics.get('Last Train Loss', 'N/A'):.4f}" if metrics.get('Last Train Loss') is not None else 'N/A',
            'Last Val Loss': f"{metrics.get('Last Val Loss', 'N/A'):.4f}" if metrics.get('Last Val Loss') is not None else 'N/A'
        }
        
        file_exists = os.path.isfile(self.csv_log_file)
        with open(self.csv_log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    
    def run_single_test(self, run_num):
        """Execute a single test run with real-time epoch progress"""
        print(f"\nRun #{run_num}/{self.num_runs}")
        start_time = time.time()
        
        try:
            # Run main.py and capture output
            process = subprocess.Popen(
                [sys.executable, "main.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Initialize progress bar
            pbar = tqdm(
                desc=f"Run #{run_num} Progress",
                total=100,
                unit="%",
                bar_format="{l_bar}{bar}| {postfix}"
            )
            output_lines = []
            last_train_loss = last_val_loss = None
            current_epoch = total_epochs = None
            
            # Read output in real-time
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    output_lines.append(line)
                    # Extract epoch and loss information
                    epoch_match = re.match(r"Epoch (\d+)/(\d+) - Train: ([\d.]+) \| Val: ([\d.]+)", line.strip())
                    if epoch_match:
                        current_epoch = int(epoch_match.group(1))
                        total_epochs = int(epoch_match.group(2))
                        last_train_loss = float(epoch_match.group(3))
                        last_val_loss = float(epoch_match.group(4))
                        # Update progress bar postfix with epoch and losses
                        pbar.set_postfix({
                            "Epoch": f"{current_epoch}/{total_epochs}",
                            "Train Loss": f"{last_train_loss:.4f}",
                            "Val Loss": f"{last_val_loss:.4f}"
                        })
                        # Update progress based on epoch (assuming linear progress)
                        if total_epochs:
                            pbar.n = (current_epoch / total_epochs) * 100
                            pbar.refresh()
            
            pbar.close()
            process.wait(timeout=1800 - (time.time() - start_time))  # Respect 30-minute timeout
            elapsed_time = time.time() - start_time
            output = "".join(output_lines)
            
            # Extract metrics
            metrics = self.extract_metrics(output)
            self.log_metrics(run_num, metrics, elapsed_time)
            
            print(f"Run #{run_num} completed in {elapsed_time:.2f} seconds")
            
            # Print metrics with None checks
            tr_val = metrics.get('Total Return %')
            print(f"  Total Return: {tr_val:.2f}%" if tr_val is not None else "  Total Return: N/A")
            
            sr_val = metrics.get('Sharpe Ratio')
            print(f"  Sharpe Ratio: {sr_val:.4f}" if sr_val is not None else "  Sharpe Ratio: N/A")
            
            md_val = metrics.get('Max Drawdown %')
            print(f"  Max Drawdown: {md_val:.2f}%" if md_val is not None else "  Max Drawdown: N/A")
            
            tl_val = metrics.get('Last Train Loss')
            print(f"  Last Train Loss: {tl_val:.4f}" if tl_val is not None else "  Last Train Loss: N/A")
            
            vl_val = metrics.get('Last Val Loss')
            print(f"  Last Val Loss: {vl_val:.4f}" if vl_val is not None else "  Last Val Loss: N/A")
            
            return True
            
        except subprocess.TimeoutExpired:
            print(f"Run #{run_num} timed out")
            with open(self.txt_log_file, 'a') as f:
                f.write(f"\nRun #{run_num} - TIMEOUT\n")
            with open(self.csv_log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['Run', 'Timestamp', 'Execution Time (s)', 'Total Return %', 'Sharpe Ratio', 'Max Drawdown %', 'Last Train Loss', 'Last Val Loss'])
                if not os.path.isfile(self.csv_log_file):
                    writer.writeheader()
                writer.writerow({'Run': run_num, 'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Execution Time (s)': 'TIMEOUT', 'Total Return %': 'N/A', 'Sharpe Ratio': 'N/A', 'Max Drawdown %': 'N/A', 'Last Train Loss': 'N/A', 'Last Val Loss': 'N/A'})
            pbar.close()
            return False
            
        except Exception as e:
            print(f"Run #{run_num} failed: {str(e)}")
            with open(self.txt_log_file, 'a') as f:
                f.write(f"\nRun #{run_num} - ERROR: {str(e)}\n")
            with open(self.csv_log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['Run', 'Timestamp', 'Execution Time (s)', 'Total Return %', 'Sharpe Ratio', 'Max Drawdown %', 'Last Train Loss', 'Last Val Loss'])
                if not os.path.isfile(self.csv_log_file):
                    writer.writeheader()
                writer.writerow({'Run': run_num, 'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'Execution Time (s)': f'ERROR: {str(e)}', 'Total Return %': 'N/A', 'Sharpe Ratio': 'N/A', 'Max Drawdown %': 'N/A', 'Last Train Loss': 'N/A', 'Last Val Loss': 'N/A'})
            pbar.close()
            return False
    
    def run_all_tests(self):
        """Execute all test runs"""
        print("\nAUTOMATED TESTING")
        print(f"Running {self.num_runs} tests")
        print(f"Logging to: {self.txt_log_file} and {self.csv_log_file}")
        
        # Initialize TXT file
        with open(self.txt_log_file, 'a') as f:
            f.write("\n" + "="*40 + "\n")
            f.write(f"TEST SESSION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Runs: {self.num_runs}\n")
        
        # Initialize CSV file with headers if it doesn't exist
        if not os.path.isfile(self.csv_log_file):
            with open(self.csv_log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['Run', 'Timestamp', 'Execution Time (s)', 'Total Return %', 'Sharpe Ratio', 'Max Drawdown %', 'Last Train Loss', 'Last Val Loss'])
                writer.writeheader()
        
        print(f"Test session started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        successful_runs = 0
        for run_num in range(1, self.num_runs + 1):
            success = self.run_single_test(run_num)
            if success:
                successful_runs += 1
            time.sleep(1)  # Brief pause between runs
        
        print("\nTESTING COMPLETE")
        print(f"Successful runs: {successful_runs}/{self.num_runs}")

if __name__ == "__main__":
    NUM_RUNS = 1
    
    if not os.path.exists("main.py"):
        print("Error: main.py not found!")
        sys.exit(1)
    
    tester = AutomatedTester(num_runs=NUM_RUNS)
    tester.run_all_tests()
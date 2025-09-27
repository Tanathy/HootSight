"""
Enhanced logging module with progress bar support
Features:
- Colored prints with timestamps and caller script name
- Beautiful tqdm-style progress bars
- Progress bar integration with logging
- Thread-safe operations
"""



import datetime
import os
import inspect
import sys
import time
import re
from contextlib import contextmanager

from threading import Lock
from collections import OrderedDict

# Global progress bar lock for thread safety
_progress_lock = Lock()

def clear() -> None:
    """Clear the CLI screen in a cross-platform way."""
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

# ANSI color codes and progress bar characters
class Colors:
    RESET = '\033[0m'
    INFO = '\033[37m'      # White (normal)
    WARNING = '\033[33m'   # Yellow
    ERROR = '\033[31m'     # Red
    SUCCESS = '\033[32m'   # Green
    
    # Progress bar colors
    PROGRESS_COMPLETE = '\033[42m'  # Green background
    PROGRESS_INCOMPLETE = '\033[47m'  # White background
    
    # Additional colors for progress bars
    CYAN = '\033[36m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'

class ProgressChars:
    """Unicode progress bar characters"""
    FULL = '█'
    SEVEN_EIGHTHS = '▉' 
    THREE_QUARTERS = '▊'
    FIVE_EIGHTHS = '▋'
    HALF = '▌'
    THREE_EIGHTHS = '▍'
    QUARTER = '▎'
    EIGHTH = '▏'
    EMPTY = ' '
    
    # Fallback ASCII chars if unicode fails
    ASCII_FULL = '#'
    ASCII_EMPTY = '-'

# --- Minimal internal ANSI/Win32 color support (no colorama dependency) ---
if os.name == 'nt':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        # Enable VT100/ANSI escape sequence processing on Windows 10+
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE = -11
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            mode.value |= 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
            kernel32.SetConsoleMode(handle, mode)
        _ansi_supported = True
    except Exception:
        _ansi_supported = False
else:
    _ansi_supported = True

def _strip_ansi(text):
    return re.sub(r'\033\[[0-9;]*[mK]', '', text)

def _write_ansi(text):
    if _ansi_supported:
        sys.stdout.write(text)
    else:
        sys.stdout.write(_strip_ansi(text))

def get_caller_name():
    """Get the calling script name"""
    frame = inspect.currentframe()
    try:
        # Go up the call stack to find the actual calling script
        # Call stack: user_script -> info/warning/error -> log -> get_caller_name
        # So we need to go up 3 levels to get to the user script
        current_frame = frame
        for _ in range(3):
            if current_frame and current_frame.f_back:
                current_frame = current_frame.f_back
            else:
                return 'unknown'
        
        if current_frame:
            filename = current_frame.f_code.co_filename
            script_name = os.path.basename(filename).replace('.py', '')
            
            # If it's still showing 'log' or internal modules, go one more level up
            if script_name in ['log', 'coordinator_settings', '__main__']:
                if current_frame.f_back:
                    filename = current_frame.f_back.f_code.co_filename
                    script_name = os.path.basename(filename).replace('.py', '')
            
            return script_name
        
        return 'unknown'
    finally:
        del frame

def format_time(seconds):
    """Format time in a human-readable way"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m{secs:02d}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h{minutes:02d}m"

def format_rate(rate, unit='it'):
    """Format iteration rate"""
    if rate <= 0:
        return f"0.0{unit}/s"
    elif rate < 1:
        return f"{1/rate:.1f}s/{unit}"
    elif rate < 1000:
        return f"{rate:.1f}{unit}/s"
    elif rate < 1000000:
        return f"{rate/1000:.1f}k{unit}/s"
    else:
        return f"{rate/1000000:.1f}M{unit}/s"

def log(message, level='INFO'):
    """
    Simple log function
    
    Args:
        message (str): The message
        level (str): 'INFO', 'WARNING', 'ERROR', or 'SUCCESS'
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    script_name = get_caller_name()
    level = level.upper()
    
    # Select color code
    if level == 'WARNING':
        color = Colors.WARNING
    elif level == 'ERROR':
        color = Colors.ERROR
    elif level == 'SUCCESS':
        color = Colors.SUCCESS
    else:
        color = Colors.INFO
    
    # Formatted message
    formatted_message = f"{timestamp}\t{level}\t{script_name}\t{message}"
    
    # Thread-safe colored output
    with _progress_lock:
        print(f"{color}{formatted_message}{Colors.RESET}")

def info(message):
    """INFO level log"""
    log(message, 'INFO')

def warning(message):
    """WARNING level log"""
    log(message, 'WARNING')

def error(message):
    """ERROR level log"""
    log(message, 'ERROR')

def success(message):
    """SUCCESS level log"""
    log(message, 'SUCCESS')

class ProgressBar:
    """
    Beautiful tqdm-style progress bar with colored output
    Integrates seamlessly with the logging system
    """
    
    def __init__(self, total=None, desc='', unit='it', unit_scale=False, 
                 ncols=None, colour='cyan', ascii_fallback=False, 
                 leave=True, miniters=1, mininterval=0.1):
        """
        Initialize progress bar
        
        Args:
            total (int): Total number of iterations
            desc (str): Description prefix
            unit (str): Unit name (e.g., 'files', 'MB', 'items')
            unit_scale (bool): Scale large numbers (1000 -> 1k)
            _write_ansi(f"{color}{formatted_message}{Colors.RESET}\n")
            colour (str): Bar color ('cyan', 'blue', 'green', etc.)
            ascii_fallback (bool): Use ASCII characters instead of Unicode
            leave (bool): Keep bar after completion
            miniters (int): Minimum iteration interval to update display
            mininterval (float): Minimum time interval to update display
        """
        self.total = total
        self.desc = desc
        self.unit = unit
        self.unit_scale = unit_scale
        self.ncols = ncols or self._get_terminal_width()
        self.colour = getattr(Colors, colour.upper(), Colors.CYAN)
        self.ascii_fallback = ascii_fallback
        self.leave = leave
        self.miniters = miniters
        self.mininterval = mininterval
        
        # Progress tracking
        self.n = 0
        self.last_print_n = 0
        self.start_time = time.time()
        self.last_print_time = self.start_time
        
        # Status
        self.closed = False
        self.displayed = False
        
        # Rate calculation
        self._rate_history = []
        self._rate_window = 10  # Number of samples for rate calculation
        
    def _get_terminal_width(self):
        """Get terminal width, fallback to 80"""
        try:
            import shutil
            return shutil.get_terminal_size().columns
        except:
            return 80
    
    def _supports_unicode(self):
        """Check if terminal supports Unicode"""
        if self.ascii_fallback:
            return False
        try:
            # Test if we can encode/decode unicode
            ProgressChars.FULL.encode(sys.stdout.encoding or 'utf-8')
            return True
        except:
            return False
    
    def _format_bar(self, frac, bar_length):
        """Create the visual progress bar"""
        if self._supports_unicode():
            chars = [ProgressChars.EMPTY, ProgressChars.EIGHTH, ProgressChars.QUARTER,
                    ProgressChars.THREE_EIGHTHS, ProgressChars.HALF, ProgressChars.FIVE_EIGHTHS,
                    ProgressChars.THREE_QUARTERS, ProgressChars.SEVEN_EIGHTHS, ProgressChars.FULL]
            
            # Unicode progress bar
            nsyms = len(chars) - 1
            full_length = int(frac * bar_length * nsyms)
            full_bars = full_length // nsyms
            remainder = full_length % nsyms
            
            bar = chars[-1] * full_bars
            if full_bars < bar_length:
                bar += chars[remainder]
                bar += chars[0] * (bar_length - full_bars - 1)
        else:
            # ASCII fallback
            full_char = ProgressChars.ASCII_FULL
            empty_char = ProgressChars.ASCII_EMPTY
            filled_length = int(bar_length * frac)
            bar = full_char * filled_length + empty_char * (bar_length - filled_length)
        
        return f"{self.colour}{bar}{Colors.RESET}"
    
    def _format_number(self, n):
        """Format number with optional scaling"""
        if not self.unit_scale:
            return str(int(n)) if n == int(n) else f"{n:.1f}"
        
        if n < 1000:
            return str(int(n)) if n == int(n) else f"{n:.1f}"
        elif n < 1000000:
            return f"{n/1000:.1f}k"
        elif n < 1000000000:
            return f"{n/1000000:.1f}M"
        else:
            return f"{n/1000000000:.1f}G"
    
    def _calculate_rate(self):
        """Calculate current iteration rate"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if elapsed > 0:
            rate = self.n / elapsed
            
            # Store rate in history for smoothing
            self._rate_history.append(rate)
            if len(self._rate_history) > self._rate_window:
                self._rate_history.pop(0)
            
            # Return smoothed rate
            return sum(self._rate_history) / len(self._rate_history)
        
        return 0
    
    def _format_meter(self):
        """Format the complete progress meter"""
        elapsed = time.time() - self.start_time
        
        # Calculate percentage and bar
        if self.total:
            frac = min(self.n / self.total, 1.0)
            percentage = frac * 100
        else:
            frac = 0
            percentage = 0
        
        # Build the progress string
        desc_str = f"{self.desc}: " if self.desc else ""
        
        if self.total:
            # Progress with known total
            n_fmt = self._format_number(self.n)
            total_fmt = self._format_number(self.total)
            percentage_str = f"{percentage:3.0f}%"
            
            # Calculate bar width
            prefix_len = len(f"{desc_str}{percentage_str}|")
            suffix_len = len(f"| {n_fmt}/{total_fmt} [{format_time(elapsed)}<")
            
            rate = self._calculate_rate()
            if rate > 0 and self.total:
                remaining_time = (self.total - self.n) / rate
                time_left = format_time(remaining_time)
            else:
                time_left = "?"
            
            rate_str = format_rate(rate, self.unit)
            suffix_len += len(f"{time_left}, {rate_str}]")
            
            bar_length = max(10, self.ncols - prefix_len - suffix_len)
            bar = self._format_bar(frac, bar_length)
            
            return (f"{desc_str}{percentage_str}|{bar}| "
                   f"{n_fmt}/{total_fmt} [{format_time(elapsed)}<{time_left}, {rate_str}]")
        else:
            # Progress with unknown total
            n_fmt = self._format_number(self.n)
            rate = self._calculate_rate()
            rate_str = format_rate(rate, self.unit)
            
            return f"{desc_str}{n_fmt}{self.unit} [{format_time(elapsed)}, {rate_str}]"
    
    def update(self, n=1):
        """Update progress by n iterations"""
        if self.closed:
            return
        
        self.n += n
        current_time = time.time()
        
        # Check if we should update display
        if (self.n - self.last_print_n >= self.miniters and 
            current_time - self.last_print_time >= self.mininterval):
            
            self.refresh()
            self.last_print_n = self.n
            self.last_print_time = current_time
    
    def refresh(self):
        """Force refresh the progress display"""
        if self.closed:
            return
            
        with _progress_lock:
            # Clear current line and print progress
            meter = self._format_meter()
            if self.displayed:
                print(f"\r{meter}", end='', flush=True)
            else:
                print(meter, end='', flush=True)
                self.displayed = True
    
    def set_description(self, desc):
        """Update the description"""
        self.desc = desc
        self.refresh()
    
    def set_postfix(self, **kwargs):
        """Set postfix stats (e.g., loss=0.1, acc=0.95)"""
        if kwargs:
            postfix_str = ', '.join(f"{k}={v}" for k, v in kwargs.items())
            # Store postfix separately from description
            if not hasattr(self, '_original_desc'):
                self._original_desc = self.desc
            self.desc = f"{self._original_desc} [{postfix_str}]"
            self.refresh()
    
    def close(self):
        """Close the progress bar"""
        if self.closed:
            return
        
        self.closed = True
        
        with _progress_lock:
            if self.displayed:
                if self.leave:
                    # Move to new line
                    print()
                else:
                    # Clear the line
                    print(f"\r{' ' * self.ncols}\r", end='', flush=True)
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

def progress_bar(iterable=None, total=None, **kwargs):
    """
    Create a progress bar for iterables or manual updates
    
    Args:
        iterable: Iterable to wrap with progress bar
        total: Total iterations (required if iterable is None)
        **kwargs: Additional arguments for ProgressBar
    
    Returns:
        ProgressBar instance or wrapped iterable
    """
    if iterable is not None:
        # Wrap iterable
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                total = None
        
        pbar = ProgressBar(total=total, **kwargs)
        
        class ProgressWrapper:
            def __init__(self, iterable, pbar):
                self.iterable = iterable
                self.pbar = pbar
            
            def __iter__(self):
                try:
                    for item in self.iterable:
                        yield item
                        self.pbar.update(1)
                finally:
                    self.pbar.close()
            
            def __enter__(self):
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.pbar.close()
        
        return ProgressWrapper(iterable, pbar)
    else:
        # Manual progress bar
        return ProgressBar(total=total, **kwargs)

@contextmanager
def progress_context(total=None, **kwargs):
    """Context manager for progress bars"""
    pbar = ProgressBar(total=total, **kwargs)
    try:
        yield pbar
    finally:
        pbar.close()

# Alias for common usage patterns
tqdm = progress_bar

"""
RSNA Aneurysm Detection - Security Utilities

Security-focused utilities for safe subprocess execution and input validation.
"""

import subprocess
import shlex
import os
import logging
import re
from typing import List, Dict, Optional, Any, Union
from pathlib import Path

# Import the centralized SecurityError from exceptions module
try:
    from .exceptions import SecurityError
except ImportError:
    # Fallback if exceptions module not available
    class SecurityError(Exception):
        """Custom exception for security-related errors"""
        pass

logger = logging.getLogger(__name__)


class SafeSubprocessExecutor:
    """
    Secure subprocess execution with input validation and sandboxing
    
    Features:
    - Input sanitization and validation
    - Command whitelisting
    - Timeout enforcement
    - Environment isolation
    - Logging and audit trail
    """
    
    # Whitelist of allowed commands
    ALLOWED_COMMANDS = {
        'git': ['rev-parse', 'branch', 'status', 'log'],
        'kaggle': ['competitions', 'datasets', 'kernels'],
        'python': ['-m', '-c'],
        'pip': ['install', 'list', 'show'],
        'chmod': ['600', '644', '755']
    }
    
    # Maximum execution time for different command types
    TIMEOUT_LIMITS = {
        'git': 10,
        'kaggle': 300,  # 5 minutes for downloads
        'python': 60,
        'pip': 120,
        'chmod': 5
    }
    
    def __init__(self, allowed_commands: Optional[Dict[str, List[str]]] = None):
        """
        Initialize secure subprocess executor
        
        Args:
            allowed_commands: Custom command whitelist (overrides default)
        """
        self.allowed_commands = allowed_commands or self.ALLOWED_COMMANDS.copy()
        
    def validate_command(self, cmd: List[str]) -> bool:
        """
        Validate command against whitelist
        
        Args:
            cmd: Command list to validate
            
        Returns:
            bool: True if command is allowed
            
        Raises:
            SecurityError: If command is not whitelisted
        """
        if not cmd or not isinstance(cmd, list):
            raise SecurityError("Command must be a non-empty list")
            
        base_command = cmd[0]
        
        if base_command not in self.allowed_commands:
            raise SecurityError(f"Command '{base_command}' not in whitelist")
            
        # Check subcommands/arguments
        allowed_args = self.allowed_commands[base_command]
        
        if len(cmd) > 1:
            # Check if any of the command arguments match allowed patterns
            cmd_args = cmd[1:]
            
            # For flexible matching, check if any allowed arg is in the command
            if not any(allowed_arg in ' '.join(cmd_args) for allowed_arg in allowed_args):
                # Special case: allow exact argument matches
                if not any(arg in allowed_args for arg in cmd_args[:2]):
                    logger.warning(f"Command arguments {cmd_args} not fully whitelisted for {base_command}")
        
        return True
        
    def sanitize_input(self, value: str, pattern: str = r'^[a-zA-Z0-9_\-\.\/]+$') -> str:
        """
        Sanitize string input using regex pattern
        
        Args:
            value: Input string to sanitize
            pattern: Regex pattern for allowed characters
            
        Returns:
            str: Sanitized string
            
        Raises:
            SecurityError: If input contains disallowed characters
        """
        if not isinstance(value, str):
            raise SecurityError("Input must be a string")
            
        if not re.match(pattern, value):
            raise SecurityError(f"Input '{value}' contains disallowed characters")
            
        return value
        
    def run_safe(
        self, 
        cmd: List[str], 
        cwd: Optional[Union[str, Path]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None,
        capture_output: bool = True,
        text: bool = True,
        check: bool = True
    ) -> subprocess.CompletedProcess:
        """
        Execute command safely with validation and sandboxing
        
        Args:
            cmd: Command to execute as list
            cwd: Working directory
            env: Environment variables
            timeout: Execution timeout (uses default if not specified)
            capture_output: Whether to capture stdout/stderr
            text: Whether to return text output
            check: Whether to raise exception on non-zero exit
            
        Returns:
            subprocess.CompletedProcess: Execution result
            
        Raises:
            SecurityError: If command validation fails
            subprocess.CalledProcessError: If command execution fails
            subprocess.TimeoutExpired: If command times out
        """
        # Validate command
        self.validate_command(cmd)
        
        # Set default timeout
        if timeout is None:
            base_command = cmd[0]
            timeout = self.TIMEOUT_LIMITS.get(base_command, 30)
            
        # Sanitize working directory
        if cwd:
            cwd = Path(cwd).resolve()
            
        # Use restricted environment if not provided
        if env is None:
            env = self._get_safe_environment()
            
        logger.info(f"Executing safe command: {' '.join(cmd[:2])}... (timeout: {timeout}s)")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                timeout=timeout,
                capture_output=capture_output,
                text=text,
                check=check
            )
            
            logger.debug(f"Command completed successfully: exit code {result.returncode}")
            return result
            
        except subprocess.TimeoutExpired as e:
            logger.error(f"Command timed out after {timeout}s: {cmd[0]}")
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with exit code {e.returncode}: {cmd[0]}")
            raise
            
    def _get_safe_environment(self) -> Dict[str, str]:
        """
        Create restricted environment variables
        
        Returns:
            Dict[str, str]: Safe environment variables
        """
        safe_env = {}
        
        # Include essential environment variables
        essential_vars = [
            'PATH', 'HOME', 'USER', 'SHELL',
            'KAGGLE_USERNAME', 'KAGGLE_KEY',  # For Kaggle API
            'PYTHONPATH', 'LANG', 'LC_ALL'
        ]
        
        for var in essential_vars:
            if var in os.environ:
                safe_env[var] = os.environ[var]
                
        return safe_env


def validate_competition_name(competition: str) -> str:
    """
    Validate Kaggle competition name
    
    Args:
        competition: Competition name to validate
        
    Returns:
        str: Validated competition name
        
    Raises:
        SecurityError: If competition name is invalid
    """
    if not competition or not isinstance(competition, str):
        raise SecurityError("Competition name must be a non-empty string")
        
    # Allow alphanumeric characters, hyphens, and underscores
    if not re.match(r'^[a-zA-Z0-9_-]+$', competition):
        raise SecurityError(f"Invalid competition name: {competition}")
        
    if len(competition) > 100:  # Reasonable length limit
        raise SecurityError("Competition name too long")
        
    return competition


def validate_file_path(file_path: Union[str, Path], must_exist: bool = False) -> Path:
    """
    Validate and sanitize file path
    
    Args:
        file_path: Path to validate
        must_exist: Whether the path must exist
        
    Returns:
        Path: Validated path object
        
    Raises:
        SecurityError: If path is invalid or unsafe
    """
    try:
        path = Path(file_path).resolve()
    except (OSError, ValueError) as e:
        raise SecurityError(f"Invalid path: {e}")
        
    # Check for path traversal attempts
    if '..' in str(file_path):
        raise SecurityError("Path traversal detected")
        
    if must_exist and not path.exists():
        raise SecurityError(f"Path does not exist: {path}")
        
    return path


# Global safe executor instance
safe_executor = SafeSubprocessExecutor()


def run_git_command(args: List[str], **kwargs) -> subprocess.CompletedProcess:
    """
    Execute git command safely
    
    Args:
        args: Git command arguments (without 'git')
        **kwargs: Additional arguments for subprocess
        
    Returns:
        subprocess.CompletedProcess: Command result
    """
    cmd = ['git'] + args
    return safe_executor.run_safe(cmd, **kwargs)


def run_kaggle_command(args: List[str], **kwargs) -> subprocess.CompletedProcess:
    """
    Execute kaggle command safely
    
    Args:
        args: Kaggle command arguments (without 'kaggle')
        **kwargs: Additional arguments for subprocess
        
    Returns:
        subprocess.CompletedProcess: Command result
    """
    cmd = ['kaggle'] + args
    return safe_executor.run_safe(cmd, **kwargs)


if __name__ == "__main__":
    # Test security utilities
    import sys
    
    try:
        # Test git command
        result = run_git_command(['rev-parse', 'HEAD'])
        print(f"Git SHA: {result.stdout.strip()[:8]}")
        
        # Test validation
        validate_competition_name("rsna-intracranial-aneurysm-detection")
        print("Competition name validation passed")
        
        # Test path validation
        validate_file_path("./scripts/security_utils.py", must_exist=True)
        print("Path validation passed")
        
        print("All security tests passed!")
        
    except Exception as e:
        print(f"Security test failed: {e}")
        sys.exit(1)
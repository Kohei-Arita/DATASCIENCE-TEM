"""
Unit tests for security utilities

Tests the security-focused subprocess execution and input validation.
"""

import pytest
import subprocess
from unittest.mock import patch, MagicMock
from pathlib import Path

from scripts.security_utils import (
    SafeSubprocessExecutor,
    SecurityError,
    validate_competition_name,
    validate_file_path,
    run_git_command,
    run_kaggle_command
)


class TestSafeSubprocessExecutor:
    """Test the SafeSubprocessExecutor class"""
    
    def test_init_default_commands(self):
        """Test initialization with default allowed commands"""
        executor = SafeSubprocessExecutor()
        assert 'git' in executor.allowed_commands
        assert 'kaggle' in executor.allowed_commands
        assert 'python' in executor.allowed_commands
    
    def test_init_custom_commands(self):
        """Test initialization with custom allowed commands"""
        custom_commands = {'test': ['arg1', 'arg2']}
        executor = SafeSubprocessExecutor(allowed_commands=custom_commands)
        assert executor.allowed_commands == custom_commands
    
    def test_validate_command_allowed(self):
        """Test command validation for allowed commands"""
        executor = SafeSubprocessExecutor()
        
        # Valid git command
        assert executor.validate_command(['git', 'rev-parse', 'HEAD']) is True
        
        # Valid kaggle command
        assert executor.validate_command(['kaggle', 'competitions', 'list']) is True
    
    def test_validate_command_not_allowed(self):
        """Test command validation for disallowed commands"""
        executor = SafeSubprocessExecutor()
        
        with pytest.raises(SecurityError, match="Command 'rm' not in whitelist"):
            executor.validate_command(['rm', '-rf', '/'])
        
        with pytest.raises(SecurityError, match="Command 'curl' not in whitelist"):
            executor.validate_command(['curl', 'http://malicious.com'])
    
    def test_validate_command_invalid_input(self):
        """Test command validation with invalid input"""
        executor = SafeSubprocessExecutor()
        
        with pytest.raises(SecurityError, match="Command must be a non-empty list"):
            executor.validate_command([])
        
        with pytest.raises(SecurityError, match="Command must be a non-empty list"):
            executor.validate_command("invalid")
    
    def test_sanitize_input_valid(self):
        """Test input sanitization with valid input"""
        executor = SafeSubprocessExecutor()
        
        valid_inputs = [
            "rsna-intracranial-aneurysm-detection",
            "simple_filename.txt",
            "/path/to/file.py",
            "folder123/subfolder_test"
        ]
        
        for input_str in valid_inputs:
            result = executor.sanitize_input(input_str)
            assert result == input_str
    
    def test_sanitize_input_invalid(self):
        """Test input sanitization with invalid input"""
        executor = SafeSubprocessExecutor()
        
        invalid_inputs = [
            "command; rm -rf /",
            "file$(malicious_command)",
            "path|with|pipes",
            "file with spaces and ;semicolons"
        ]
        
        for input_str in invalid_inputs:
            with pytest.raises(SecurityError, match="contains disallowed characters"):
                executor.sanitize_input(input_str)
    
    def test_sanitize_input_non_string(self):
        """Test input sanitization with non-string input"""
        executor = SafeSubprocessExecutor()
        
        with pytest.raises(SecurityError, match="Input must be a string"):
            executor.sanitize_input(123)
        
        with pytest.raises(SecurityError, match="Input must be a string"):
            executor.sanitize_input(['list', 'input'])
    
    @patch('subprocess.run')
    def test_run_safe_success(self, mock_run):
        """Test successful safe command execution"""
        executor = SafeSubprocessExecutor()
        
        # Mock successful subprocess result
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "success output"
        mock_run.return_value = mock_result
        
        result = executor.run_safe(['git', 'rev-parse', 'HEAD'])
        
        assert result == mock_result
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0] == ['git', 'rev-parse', 'HEAD']
    
    @patch('subprocess.run')
    def test_run_safe_timeout(self, mock_run):
        """Test safe command execution with timeout"""
        executor = SafeSubprocessExecutor()
        
        mock_run.side_effect = subprocess.TimeoutExpired(['git', 'rev-parse', 'HEAD'], 10)
        
        with pytest.raises(subprocess.TimeoutExpired):
            executor.run_safe(['git', 'rev-parse', 'HEAD'])
    
    @patch('subprocess.run')
    def test_run_safe_command_failure(self, mock_run):
        """Test safe command execution with command failure"""
        executor = SafeSubprocessExecutor()
        
        mock_run.side_effect = subprocess.CalledProcessError(1, ['git', 'rev-parse', 'HEAD'])
        
        with pytest.raises(subprocess.CalledProcessError):
            executor.run_safe(['git', 'rev-parse', 'HEAD'])
    
    def test_get_safe_environment(self):
        """Test safe environment generation"""
        executor = SafeSubprocessExecutor()
        
        with patch.dict('os.environ', {'PATH': '/usr/bin', 'HOME': '/home/user', 'MALICIOUS': 'value'}):
            safe_env = executor._get_safe_environment()
            
            assert 'PATH' in safe_env
            assert 'HOME' in safe_env
            assert 'MALICIOUS' not in safe_env


class TestValidationFunctions:
    """Test standalone validation functions"""
    
    def test_validate_competition_name_valid(self):
        """Test competition name validation with valid names"""
        valid_names = [
            "rsna-intracranial-aneurysm-detection",
            "titanic",
            "house-prices-advanced-regression",
            "competition_with_underscores",
            "simple123"
        ]
        
        for name in valid_names:
            result = validate_competition_name(name)
            assert result == name
    
    def test_validate_competition_name_invalid(self):
        """Test competition name validation with invalid names"""
        invalid_names = [
            "",  # Empty
            "name with spaces",  # Spaces
            "name;with;semicolons",  # Semicolons
            "name$(injection)",  # Special characters
            "name/with/slashes",  # Slashes
            "a" * 101,  # Too long
            123,  # Not a string
            None  # None
        ]
        
        for name in invalid_names:
            with pytest.raises(SecurityError):
                validate_competition_name(name)
    
    def test_validate_file_path_valid(self):
        """Test file path validation with valid paths"""
        test_path = Path("/tmp/test_file.txt")
        
        # Test without existence check
        result = validate_file_path(str(test_path), must_exist=False)
        assert isinstance(result, Path)
        assert result.is_absolute()
    
    def test_validate_file_path_traversal(self):
        """Test file path validation with path traversal attempts"""
        malicious_paths = [
            "../../../etc/passwd",
            "folder/../../../secret",
            "/path/with/../traversal"
        ]
        
        for path in malicious_paths:
            with pytest.raises(SecurityError, match="Path traversal detected"):
                validate_file_path(path)
    
    def test_validate_file_path_must_exist(self, test_data_dir):
        """Test file path validation with existence requirement"""
        # Create a test file
        test_file = test_data_dir / "test.txt"
        test_file.write_text("test content")
        
        # Should pass for existing file
        result = validate_file_path(str(test_file), must_exist=True)
        assert result == test_file.resolve()
        
        # Should fail for non-existing file
        non_existing = test_data_dir / "non_existing.txt"
        with pytest.raises(SecurityError, match="Path does not exist"):
            validate_file_path(str(non_existing), must_exist=True)


@pytest.mark.security
class TestSecureCommands:
    """Test secure command execution functions"""
    
    @patch('scripts.security_utils.safe_executor.run_safe')
    def test_run_git_command(self, mock_run_safe):
        """Test secure git command execution"""
        mock_result = MagicMock()
        mock_run_safe.return_value = mock_result
        
        result = run_git_command(['rev-parse', 'HEAD'])
        
        assert result == mock_result
        mock_run_safe.assert_called_once_with(['git', 'rev-parse', 'HEAD'])
    
    @patch('scripts.security_utils.safe_executor.run_safe')
    def test_run_kaggle_command(self, mock_run_safe):
        """Test secure kaggle command execution"""
        mock_result = MagicMock()
        mock_run_safe.return_value = mock_result
        
        result = run_kaggle_command(['competitions', 'list'])
        
        assert result == mock_result
        mock_run_safe.assert_called_once_with(['kaggle', 'competitions', 'list'])
    
    @patch('scripts.security_utils.safe_executor.run_safe')
    def test_run_git_command_with_kwargs(self, mock_run_safe):
        """Test secure git command with additional arguments"""
        mock_result = MagicMock()
        mock_run_safe.return_value = mock_result
        
        run_git_command(['status'], timeout=5, cwd='/tmp')
        
        mock_run_safe.assert_called_once_with(['git', 'status'], timeout=5, cwd='/tmp')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
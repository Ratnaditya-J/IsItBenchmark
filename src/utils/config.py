"""
Configuration management for IsItBenchmark.

This module handles configuration loading, validation, and management
for the benchmark detection system.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class DetectionConfig:
    """Configuration for detection algorithms."""
    similarity_threshold: float = 0.55
    enable_semantic_matching: bool = True
    enable_fuzzy_matching: bool = True
    max_matches: int = 10
    semantic_model: str = "all-MiniLM-L6-v2"
    batch_size: int = 32


@dataclass
class DatabaseConfig:
    """Configuration for database settings."""
    database_path: Optional[str] = None
    auto_update: bool = True
    update_interval_hours: int = 24


@dataclass
class APIConfig:
    """Configuration for API server."""
    host: str = "localhost"
    port: int = 8000
    debug: bool = False
    cors_enabled: bool = True
    rate_limit_enabled: bool = True
    max_requests_per_minute: int = 100


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None


class Config:
    """
    Main configuration manager for IsItBenchmark.
    
    Handles loading configuration from files, environment variables,
    and provides default values.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize with defaults
        self.detection = DetectionConfig()
        self.database = DatabaseConfig()
        self.api = APIConfig()
        self.logging = LoggingConfig()
        
        # Load configuration from file if provided
        if config_path:
            self.load_from_file(config_path)
        
        # Override with environment variables
        self.load_from_env()
        
        # Setup logging
        self._setup_logging()
    
    def load_from_file(self, config_path: str):
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                self.logger.warning(f"Config file not found: {config_path}")
                return
            
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configuration sections
            if 'detection' in config_data:
                self._update_config(self.detection, config_data['detection'])
            
            if 'database' in config_data:
                self._update_config(self.database, config_data['database'])
            
            if 'api' in config_data:
                self._update_config(self.api, config_data['api'])
            
            if 'logging' in config_data:
                self._update_config(self.logging, config_data['logging'])
            
            self.logger.info(f"Configuration loaded from: {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {str(e)}")
    
    def load_from_env(self):
        """Load configuration from environment variables."""
        # Detection settings
        if os.getenv('ISITBENCHMARK_SIMILARITY_THRESHOLD'):
            self.detection.similarity_threshold = float(os.getenv('ISITBENCHMARK_SIMILARITY_THRESHOLD'))
        
        if os.getenv('ISITBENCHMARK_SEMANTIC_MODEL'):
            self.detection.semantic_model = os.getenv('ISITBENCHMARK_SEMANTIC_MODEL')
        
        if os.getenv('ISITBENCHMARK_ENABLE_SEMANTIC'):
            self.detection.enable_semantic_matching = os.getenv('ISITBENCHMARK_ENABLE_SEMANTIC').lower() == 'true'
        
        if os.getenv('ISITBENCHMARK_ENABLE_FUZZY'):
            self.detection.enable_fuzzy_matching = os.getenv('ISITBENCHMARK_ENABLE_FUZZY').lower() == 'true'
        
        # Database settings
        if os.getenv('ISITBENCHMARK_DATABASE_PATH'):
            self.database.database_path = os.getenv('ISITBENCHMARK_DATABASE_PATH')
        
        # API settings
        if os.getenv('ISITBENCHMARK_HOST'):
            self.api.host = os.getenv('ISITBENCHMARK_HOST')
        
        if os.getenv('ISITBENCHMARK_PORT'):
            self.api.port = int(os.getenv('ISITBENCHMARK_PORT'))
        
        if os.getenv('ISITBENCHMARK_DEBUG'):
            self.api.debug = os.getenv('ISITBENCHMARK_DEBUG').lower() == 'true'
        
        # Logging settings
        if os.getenv('ISITBENCHMARK_LOG_LEVEL'):
            self.logging.level = os.getenv('ISITBENCHMARK_LOG_LEVEL')
        
        if os.getenv('ISITBENCHMARK_LOG_FILE'):
            self.logging.file_path = os.getenv('ISITBENCHMARK_LOG_FILE')
    
    def _update_config(self, config_obj, config_dict: Dict[str, Any]):
        """Update configuration object with dictionary values."""
        for key, value in config_dict.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
            else:
                self.logger.warning(f"Unknown configuration key: {key}")
    
    def _setup_logging(self):
        """Setup logging based on configuration."""
        # Convert string level to logging constant
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL,
        }
        
        log_level = level_map.get(self.logging.level.upper(), logging.INFO)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format=self.logging.format,
            filename=self.logging.file_path,
        )
    
    def save_to_file(self, config_path: str):
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save configuration file
        """
        try:
            config_data = {
                'detection': asdict(self.detection),
                'database': asdict(self.database),
                'api': asdict(self.api),
                'logging': asdict(self.logging),
            }
            
            config_file = Path(config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            self.logger.info(f"Configuration saved to: {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save config to {config_path}: {str(e)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'detection': asdict(self.detection),
            'database': asdict(self.database),
            'api': asdict(self.api),
            'logging': asdict(self.logging),
        }
    
    def validate(self) -> bool:
        """
        Validate configuration values.
        
        Returns:
            True if configuration is valid
        """
        try:
            # Validate detection config
            if not 0.0 <= self.detection.similarity_threshold <= 1.0:
                raise ValueError("similarity_threshold must be between 0.0 and 1.0")
            
            if self.detection.max_matches <= 0:
                raise ValueError("max_matches must be positive")
            
            if self.detection.batch_size <= 0:
                raise ValueError("batch_size must be positive")
            
            # Validate API config
            if not 1 <= self.api.port <= 65535:
                raise ValueError("port must be between 1 and 65535")
            
            if self.api.max_requests_per_minute <= 0:
                raise ValueError("max_requests_per_minute must be positive")
            
            return True
            
        except ValueError as e:
            self.logger.error(f"Configuration validation failed: {str(e)}")
            return False

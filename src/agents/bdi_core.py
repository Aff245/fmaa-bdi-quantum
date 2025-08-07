#!/usr/bin/env python3
"""
FMAA BDI Agent Core - GitHub Actions Native Implementation
Quantum-enhanced Belief-Desire-Intention architecture
Optimized for GitHub Actions runner constraints
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

# Local imports
from agents.belief.belief_system import BeliefSystem
from agents.desire.desire_system import DesireSystem
from agents.intention.intention_system import IntentionSystem
from quantum.quantum_engine import QuantumEngine

class MasterBDIAgent:
    """
    Master BDI Agent optimized for GitHub Actions execution
    Integrates quantum computing for enhanced decision making
    """
    
    def __init__(self, config_path: Optional[str] = None, test_mode: bool = False):
        self.test_mode = test_mode
        self.config = self._load_config(config_path)
        self.setup_logging()
        
        # Initialize core systems
        self.belief_system = BeliefSystem(self.config, test_mode)
        self.desire_system = DesireSystem(self.config, test_mode)
        self.intention_system = IntentionSystem(self.config, test_mode)
        
        # Initialize quantum engine if available
        self.quantum_engine = None
        if self.config.get('quantum', {}).get('enabled', True):
            try:
                self.quantum_engine = QuantumEngine(self.config.get('quantum', {}))
                self.logger.info("✅ Quantum engine initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Quantum engine initialization failed: {e}")
                self.logger.info("🔄 Falling back to classical processing")
        
        # Agent state
        self.is_running = False
        self.cycle_count = 0
        self.start_time = None
        self.performance_metrics = []
        
        self.logger.info("🤖 Master BDI Agent initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration with GitHub Actions environment detection"""
        default_config = {
            'environment': 'github_actions' if self._is_github_actions() else 'local',
            'max_cycles': 1 if self._is_github_actions() else 0,
            'cycle_interval': 30,  # Shorter for GitHub Actions
            'quantum': {
                'enabled': True,
                'backend': 'qasm_simulator',
                'max_qubits': 12,  # Conservative for memory limits
                'shots': 1024
            },
            'performance': {
                'enable_profiling': True,
                'memory_limit_gb': 14,  # GitHub Actions limit
                'timeout_minutes': 30
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
                    self.logger.info(f"✅ Loaded configuration from {config_path}")
            except Exception as e:
                print(f"⚠️ Failed to load config {config_path}: {e}, using defaults")
        
        return default_config
    
    def _is_github_actions(self) -> bool:
        """Detect if running in GitHub Actions environment"""
        import os
        return os.getenv('GITHUB_ACTIONS') == 'true'
    
    def setup_logging(self):
        """Setup logging optimized for GitHub Actions"""
        log_level = getattr(logging, self.config['logging']['level'])
        log_format = self.config['logging']['format']
        
        # Create logs directory
        log_dir = Path('data/logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.StreamHandler(),  # Console output for GitHub Actions
                logging.FileHandler(log_dir / f'bdi_agent_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        
        self.logger = logging.getLogger('MasterBDI')
    
    async def run_bdi_cycles(self, max_cycles: Optional[int] = None) -> Dict[str, Any]:
        """
        Run BDI cycles optimized for GitHub Actions execution
        Returns comprehensive execution report
        """
        self.is_running = True
        self.start_time = time.time()
        
        target_cycles = max_cycles or self.config.get('max_cycles', 1)
        if target_cycles == 0:  # Infinite mode not suitable for GitHub Actions
            target_cycles = 1 if self._is_github_actions() else 10
        
        self.logger.info(f"🚀 Starting BDI Agent execution - {target_cycles} cycles")
        
        execution_report = {
            'start_time': datetime.now().isoformat(),
            'environment': self.config['environment'],
            'quantum_enabled': self.quantum_engine is not None,
            'cycles': [],
            'performance_summary': {},
            'errors': []
        }
        
        try:
            for cycle in range(target_cycles):
                if not self.is_running:
                    break
                
                cycle_start = time.time()
                self.logger.info(f"🔄 Starting BDI Cycle #{cycle + 1}/{target_cycles}")
                
                try:
                    cycle_result = await self._execute_bdi_cycle(cycle + 1)
                    cycle_result['duration_seconds'] = time.time() - cycle_start
                    execution_report['cycles'].append(cycle_result)
                    
                    self.logger.info(f"✅ Cycle #{cycle + 1} completed in {cycle_result['duration_seconds']:.2f}s")
                    
                except Exception as e:
                    error_info = {
                        'cycle': cycle + 1,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    execution_report['errors'].append(error_info)
                    self.logger.error(f"❌ Cycle #{cycle + 1} failed: {e}")
                
                # Brief pause between cycles (not needed in GitHub Actions)
                if not self._is_github_actions() and cycle < target_cycles - 1:
                    await asyncio.sleep(self.config.get('cycle_interval', 30))
                
                self.cycle_count = cycle + 1
        
        except KeyboardInterrupt:
            self.logger.info("🛑 BDI Agent execution interrupted by user")
        except Exception as e:
            self.logger.error(f"💥 Fatal error in BDI execution: {e}")
            execution_report['errors'].append({
                'fatal': True,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
        
        finally:
            await self._cleanup()
            execution_report['end_time'] = datetime.now().isoformat()
            execution_report['total_duration'] = time.time() - self.start_time
            execution_report['performance_summary'] = self._generate_performance_summary()
            
            # Save execution report
            await self._save_execution_report(execution_report)
            
            self.logger.info(f"🏁 BDI Agent execution completed: {self.cycle_count} cycles in {execution_report['total_duration']:.2f}s")
        
        return execution_report
    
    async def _execute_bdi_cycle(self, cycle_number: int) -> Dict[str, Any]:
        """Execute single BDI cycle with quantum enhancements"""
        cycle_result = {
            'cycle_number': cycle_number,
            'timestamp': datetime.now().isoformat(),
            'beliefs': {},
            'desires': {},
            'intentions': {},
            'quantum_results': {},
            'performance': {}
        }
        
        # Step 1: Update Beliefs (Monitoring and Data Collection)
        belief_start = time.time()
        try:
            beliefs = await self.belief_system.update_beliefs()
            cycle_result['beliefs'] = {
                'status': 'success',
                'data': beliefs,
                'count': len(beliefs.get('components', {}))
            }
            cycle_result['performance']['belief_duration'] = time.time() - belief_start
            self.logger.info(f"📊 Beliefs updated: {cycle_result['beliefs']['count']} components")
        except Exception as e:
            cycle_result['beliefs'] = {'status': 'error', 'error': str(e)}
            self.logger.error(f"❌ Belief system error: {e}")
        
        # Step 2: Generate Desires (Goals with Quantum Optimization)
        desire_start = time.time()
        try:
            desires = await self.desire_system.generate_desires(
                beliefs.get('data', {}), 
                quantum_engine=self.quantum_engine
            )
            cycle_result['desires'] = {
                'status': 'success',
                'data': desires,
                'count': len(desires)
            }
            cycle_result['performance']['desire_duration'] = time.time() - desire_start
            self.logger.info(f"🎯 Desires generated: {len(desires)} goals")
        except Exception as e:
            cycle_result['desires'] = {'status': 'error', 'error': str(e)}
            self.logger.error(f"❌ Desire system error: {e}")
        
        # Step 3: Execute Intentions (Action Planning and Execution)
        intention_start = time.time()
        try:
            intentions = await self.intention_system.execute_intentions(
                desires.get('data', []),
                beliefs.get('data', {}),
                quantum_engine=self.quantum_engine
            )
            cycle_result['intentions'] = {
                'status': 'success',
                'data': intentions,
                'count': len(intentions)
            }
            cycle_result['performance']['intention_duration'] = time.time() - intention_start
            self.logger.info(f"⚡ Intentions executed: {len(intentions)} actions")
        except Exception as e:
            cycle_result['intentions'] = {'status': 'error', 'error': str(e)}
            self.logger.error(f"❌ Intention system error: {e}")
        
        # Step 4: Quantum Processing Results (if available)
        if self.quantum_engine:
            try:
                quantum_results = await self.quantum_engine.get_processing_summary()
                cycle_result['quantum_results'] = quantum_results
                self.logger.info(f"⚛️ Quantum processing: {quantum_results.get('circuits_executed', 0)} circuits")
            except Exception as e:
                cycle_result['quantum_results'] = {'error': str(e)}
                self.logger.warning(f"⚠️ Quantum results error: {e}")
        
        return cycle_result
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary for the entire execution"""
        if not self.performance_metrics:
            return {}
        
        return {
            'total_cycles': self.cycle_count,
            'average_cycle_time': sum(m.get('total_duration', 0) for m in self.performance_metrics) / len(self.performance_metrics),
            'quantum_utilization': self.quantum_engine.get_utilization_stats() if self.quantum_engine else None,
            'memory_peak_mb': self._get_memory_usage(),
            'github_actions_optimized': self._is_github_actions()
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
    
    async def _save_execution_report(self, report: Dict[str, Any]):
        """Save execution report for GitHub Actions artifacts"""
        output_dir = Path('data/output')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f'bdi_execution_report_{timestamp}.json'
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"📄 Execution report saved: {report_path}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save execution report: {e}")
    
    async def _cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        
        if self.quantum_engine:
            await self.quantum_engine.cleanup()
        
        await self.belief_system.cleanup()
        await self.desire_system.cleanup()
        await self.intention_system.cleanup()
        
        self.logger.info("🧹 BDI Agent cleanup completed")


async def main():
    """Main entry point with CLI interface"""
    parser = argparse.ArgumentParser(description='FMAA BDI Agent - Quantum-Enhanced')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode')
    parser.add_argument('--cycles', type=int, default=1, help='Number of BDI cycles to run')
    parser.add_argument('--benchmark-mode', action='store_true', help='Run performance benchmarks')
    
    args = parser.parse_args()
    
    # Initialize BDI Agent
    agent = MasterBDIAgent(
        config_path=args.config,
        test_mode=args.test_mode
    )
    
    if args.benchmark_mode:
        # Run extended benchmarks
        print("🏃 Running performance benchmarks...")
        results = await agent.run_bdi_cycles(max_cycles=10)
    else:
        # Normal execution
        results = await agent.run_bdi_cycles(max_cycles=args.cycles)
    
    # Print summary for GitHub Actions logs
    print(f"\n{'='*50}")
    print(f"🤖 BDI Agent Execution Summary")
    print(f"{'='*50}")
    print(f"Environment: {results.get('environment')}")
    print(f"Quantum Enabled: {results.get('quantum_enabled')}")
    print(f"Cycles Completed: {len(results.get('cycles', []))}")
    print(f"Total Duration: {results.get('total_duration', 0):.2f}s")
    print(f"Errors: {len(results.get('errors', []))}")
    
    if results.get('errors'):
        print(f"\n❌ Errors encountered:")
        for error in results['errors']:
            print(f"  - Cycle {error.get('cycle', 'N/A')}: {error.get('error')}")
    
    return 0 if not results.get('errors') else 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

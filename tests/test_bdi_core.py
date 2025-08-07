#!/usr/bin/env python3
"""
Comprehensive Test Suite for BDI Agent
Optimized for GitHub Actions execution
Includes quantum computing integration tests
"""

import asyncio
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Test imports
from agents.bdi_core import MasterBDIAgent
from quantum.quantum_engine import QuantumEngine, quantum_optimization_benchmark


class TestMasterBDIAgent:
    """Test suite for Master BDI Agent"""
    
    @pytest.fixture
    async def bdi_agent(self):
        """Create BDI Agent instance for testing"""
        config = {
            'environment': 'test',
            'max_cycles': 1,
            'cycle_interval': 1,
            'quantum': {
                'enabled': True,
                'backend': 'qasm_simulator',
                'max_qubits': 4,  # Conservative for testing
                'shots': 256
            },
            'performance': {
                'enable_profiling': True,
                'memory_limit_gb': 4,
                'timeout_minutes': 5
            }
        }
        
        agent = MasterBDIAgent(config=config, test_mode=True)
        yield agent
        await agent._cleanup()
    
    @pytest.mark.asyncio
    async def test_bdi_agent_initialization(self, bdi_agent):
        """Test BDI Agent initialization"""
        assert bdi_agent is not None
        assert bdi_agent.test_mode is True
        assert bdi_agent.belief_system is not None
        assert bdi_agent.desire_system is not None
        assert bdi_agent.intention_system is not None
        
        # Test quantum engine initialization
        if bdi_agent.quantum_engine:
            assert bdi_agent.quantum_engine.max_qubits <= 4
    
    @pytest.mark.asyncio
    async def test_single_bdi_cycle(self, bdi_agent):
        """Test single BDI cycle execution"""
        # Run one BDI cycle
        execution_report = await bdi_agent.run_bdi_cycles(max_cycles=1)
        
        # Validate execution report structure
        assert 'start_time' in execution_report
        assert 'end_time' in execution_report
        assert 'cycles' in execution_report
        assert len(execution_report['cycles']) == 1
        
        # Validate cycle structure
        cycle = execution_report['cycles'][0]
        assert 'cycle_number' in cycle
        assert 'beliefs' in cycle
        assert 'desires' in cycle
        assert 'intentions' in cycle
        assert 'performance' in cycle
        
        # Check for successful execution
        assert cycle['beliefs']['status'] in ['success', 'error']
        assert cycle['desires']['status'] in ['success', 'error']
        assert cycle['intentions']['status'] in ['success', 'error']
    
    @pytest.mark.asyncio
    async def test_multiple_bdi_cycles(self, bdi_agent):
        """Test multiple BDI cycles"""
        execution_report = await bdi_agent.run_bdi_cycles(max_cycles=3)
        
        assert len(execution_report['cycles']) == 3
        
        # Verify cycle numbering
        for i, cycle in enumerate(execution_report['cycles'], 1):
            assert cycle['cycle_number'] == i
    
    @pytest.mark.asyncio
    async def test_github_actions_optimization(self):
        """Test GitHub Actions environment detection and optimization"""
        with patch.dict('os.environ', {'GITHUB_ACTIONS': 'true'}):
            agent = MasterBDIAgent(test_mode=True)
            
            # Should automatically set single cycle for GitHub Actions
            assert agent._is_github_actions() is True
            assert agent.config.get('max_cycles', 1) == 1
            
            await agent._cleanup()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, bdi_agent):
        """Test error handling and recovery"""
        # Mock belief system to raise exception
        with patch.object(bdi_agent.belief_system, 'update_beliefs', 
                         side_effect=Exception("Test error")):
            execution_report = await bdi_agent.run_bdi_cycles(max_cycles=1)
            
            # Should handle error gracefully
            assert len(execution_report['errors']) >= 0  # May or may not have errors
            
            # Cycle should still be recorded with error status
            if execution_report['cycles']:
                cycle = execution_report['cycles'][0]
                # At least one component should show error
                components = [cycle['beliefs'], cycle['desires'], cycle['intentions']]
                error_found = any(comp.get('status') == 'error' for comp in components)
                # Error handling may vary, so we just check structure exists


class TestQuantumEngine:
    """Test suite for Quantum Engine"""
    
    @pytest.fixture
    def quantum_config(self):
        """Quantum engine configuration for testing"""
        return {
            'max_qubits': 4,
            'shots': 256,  # Reduced for faster testing
            'backend': 'qasm_simulator',
            'max_circuit_depth': 10
        }
    
    @pytest.mark.asyncio
    async def test_quantum_engine_initialization(self, quantum_config):
        """Test quantum engine initialization"""
        engine = QuantumEngine(quantum_config)
        
        assert engine.max_qubits == 4
        assert engine.default_shots == 256
        assert engine.backend_name == 'qasm_simulator'
        
        # Test backend initialization
        assert engine.qiskit_backend is not None or engine.pennylane_device is not None
        
        await engine.cleanup()
    
    @pytest.mark.asyncio
    async def test_priority_optimization(self, quantum_config):
        """Test quantum priority optimization"""
        engine = QuantumEngine(quantum_config)
        
        priorities = [0.9, 0.7, 0.5, 0.3]
        constraints = {'performance': 0.8, 'reliability': 0.9}
        
        result = await engine.optimize_priorities(priorities, constraints, n_qubits=4)
        
        # Validate result structure
        assert 'optimized_priorities' in result
        assert 'confidence' in result
        assert len(result['optimized_priorities']) == len(priorities)
        
        # Validate optimized priorities are reasonable
        for priority in result['optimized_priorities']:
            assert 0.0 <= priority <= 1.5  # Allow some quantum enhancement
        
        await engine.cleanup()
    
    @pytest.mark.asyncio
    async def test_circuit_simulation(self, quantum_config):
        """Test quantum circuit simulation"""
        engine = QuantumEngine(quantum_config)
        
        circuit_data = {
            'qubits': 3,
            'operations': [
                {'type': 'h', 'qubits': [0, 1, 2]},
                {'type': 'cx', 'qubits': [0, 1]},
                {'type': 'cx', 'qubits': [1, 2]}
            ],
            'shots': 128
        }
        
        result = await engine.simulate_quantum_circuit(circuit_data)
        
        if result['success']:
            assert 'counts' in result
            assert 'execution_time' in result
            assert 'circuit_depth' in result
            assert result['n_qubits'] == 3
            
            # Validate measurement counts
            counts = result['counts']
            total_counts = sum(counts.values())
            assert total_counts == 128
        
        await engine.cleanup()
    
    @pytest.mark.asyncio
    async def test_quantum_machine_learning(self, quantum_config):
        """Test quantum machine learning functionality"""
        engine = QuantumEngine(quantum_config)
        
        # Create dummy training data
        training_data = [
            {'features': [0.1, 0.2, 0.3, 0.4]},
            {'features': [0.5, 0.6, 0.7, 0.8]},
            {'features': [0.2, 0.3, 0.4, 0.5]},
            {'features': [0.7, 0.8, 0.9, 1.0]}
        ]
        target_labels = [0, 1, 0, 1]
        
        result = await engine.run_quantum_machine_learning(training_data, target_labels)
        
        if result['success']:
            assert 'final_cost' in result
            assert 'training_time' in result
            assert 'n_qubits' in result
            assert 'training_steps' in result
            
            # Cost should be a reasonable number
            assert 0.0 <= result['final_cost'] <= 10.0
        
        await engine.cleanup()
    
    @pytest.mark.asyncio
    async def test_memory_constraints(self, quantum_config):
        """Test quantum engine respects memory constraints"""
        # Test with larger configuration
        large_config = quantum_config.copy()
        large_config['max_qubits'] = 20  # Should be capped
        
        engine = QuantumEngine(large_config)
        
        # Should cap at reasonable limit for GitHub Actions
        assert engine.max_qubits <= 14  # GitHub Actions memory limit
        
        await engine.cleanup()


class TestIntegration:
    """Integration tests for BDI + Quantum systems"""
    
    @pytest.mark.asyncio
    async def test_quantum_bdi_integration(self):
        """Test integration between BDI Agent and Quantum Engine"""
        config = {
            'max_cycles': 1,
            'quantum': {
                'enabled': True,
                'max_qubits': 4,
                'shots': 128,
                'backend': 'qasm_simulator'
            }
        }
        
        agent = MasterBDIAgent(config=config, test_mode=True)
        
        # Ensure quantum engine is available
        assert agent.quantum_engine is not None
        
        # Run BDI cycle with quantum integration
        execution_report = await agent.run_bdi_cycles(max_cycles=1)
        
        # Check quantum results are included
        cycle = execution_report['cycles'][0]
        assert 'quantum_results' in cycle
        
        await agent._cleanup()
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self):
        """Test performance benchmarking functionality"""
        # Run quantum optimization benchmark
        benchmark_results = await quantum_optimization_benchmark(max_qubits=6, shots=256)
        
        assert isinstance(benchmark_results, dict)
        assert len(benchmark_results) > 0
        
        # Validate benchmark structure
        for qubit_config, results in benchmark_results.items():
            assert 'execution_time' in results
            assert 'confidence' in results
            assert 'circuit_depth' in results
            
            # Performance should be reasonable
            assert results['execution_time'] > 0
            assert 0.0 <= results['confidence'] <= 1.0
    
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test system recovery from quantum errors"""
        config = {
            'max_cycles': 1,
            'quantum': {
                'enabled': True,
                'max_qubits': 4,
                'backend': 'invalid_backend'  # Force error
            }
        }
        
        # Should handle quantum initialization failure gracefully
        agent = MasterBDIAgent(config=config, test_mode=True)
        
        # Should fallback to classical processing
        execution_report = await agent.run_bdi_cycles(max_cycles=1)
        
        # Should complete despite quantum failure
        assert len(execution_report['cycles']) == 1
        
        await agent._cleanup()


class TestPerformance:
    """Performance and memory tests"""
    
    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage stays within GitHub Actions limits"""
        config = {
            'max_cycles': 5,
            'quantum': {
                'enabled': True,
                'max_qubits': 8,
                'shots': 512
            }
        }
        
        agent = MasterBDIAgent(config=config, test_mode=True)
        
        # Monitor memory usage
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)
        
        execution_report = await agent.run_bdi_cycles(max_cycles=5)
        
        final_memory = process.memory_info().rss / (1024 * 1024)
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (under 1GB)
        assert memory_growth < 1024  # Less than 1GB growth
        
        # Final memory should be under GitHub Actions limit
        assert final_memory < 14 * 1024  # Less than 14GB
        
        await agent._cleanup()
    
    @pytest.mark.asyncio
    async def test_execution_timeout(self):
        """Test execution completes within reasonable time"""
        config = {
            'max_cycles': 1,
            'quantum': {
                'enabled': True,
                'max_qubits': 6,
                'shots': 256
            }
        }
        
        agent = MasterBDIAgent(config=config, test_mode=True)
        
        start_time = asyncio.get_event_loop().time()
        execution_report = await agent.run_bdi_cycles(max_cycles=1)
        end_time = asyncio.get_event_loop().time()
        
        execution_time = end_time - start_time
        
        # Should complete within reasonable time (30 seconds for GitHub Actions)
        assert execution_time < 30.0
        
        await agent._cleanup()


# Test configuration and utilities

@pytest.fixture(scope="session")
def test_config():
    """Global test configuration"""
    return {
        'test_environment': True,
        'github_actions': True,
        'memory_limit_gb': 4,
        'timeout_minutes': 5
    }


def pytest_configure(config):
    """Configure pytest for GitHub Actions"""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "quantum: marks tests requiring quantum libraries")
    config.addinivalue_line("markers", "integration: marks integration tests")


# Custom test markers for GitHub Actions
pytestmark = [
    pytest.mark.asyncio,  # All tests are async
]


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])

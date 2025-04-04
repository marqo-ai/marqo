"""
Test that pydantic v2 functionality works properly
"""
import unittest
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
from typing import Dict, Optional


class TestModel(BaseModel):
    name: str
    age: int
    metadata: Optional[Dict] = None
    
    @field_validator('age')
    def validate_age(cls, v):
        if v < 0:
            raise ValueError('Age must be positive')
        return v
    
    @model_validator(mode='after')
    def validate_model(self):
        if self.name == 'test' and self.age < 18:
            raise ValueError('Test users must be adults')
        return self


class TestPydanticV2(unittest.TestCase):
    
    def test_validation_works(self):
        # Valid model
        model = TestModel(name="John", age=30)
        self.assertEqual(model.name, "John")
        self.assertEqual(model.age, 30)
        
        # Test field_validator
        with self.assertRaises(ValidationError) as context:
            TestModel(name="John", age=-5)
        
        self.assertIn("Age must be positive", str(context.exception))
        
        # Test model_validator
        with self.assertRaises(ValidationError) as context:
            TestModel(name="test", age=15)
        
        self.assertIn("Test users must be adults", str(context.exception))
    
    def test_pydantic_version(self):
        """Check that we're using pydantic v2"""
        import pydantic
        version = pydantic.__version__
        self.assertTrue(version.startswith("2."), f"Expected pydantic v2, got {version}")


if __name__ == '__main__':
    unittest.main() 
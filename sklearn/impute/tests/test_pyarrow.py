"""Test PyArrow extension array handling in impute module."""
import pytest
import numpy as np

from sklearn.impute import SimpleImputer


@pytest.mark.parametrize("strategy", ["constant", "most_frequent"])
def test_simple_imputer_pyarrow_extension_array(strategy):
    """Test SimpleImputer preserves integer dtype with PyArrow extension arrays."""
    pd = pytest.importorskip("pandas")
    pa = pytest.importorskip("pyarrow")
    
    # Create a Series with int32[pyarrow] extension dtype 
    # This is similar to what polars.to_pandas(use_pyarrow_extension_array=True) produces
    data = pa.array([1, 2, 3], type=pa.int32())
    series = pd.Series(data)
    df = pd.DataFrame({"a": series})
    
    # Verify our test data uses PyArrow extension arrays
    assert str(df["a"].dtype).endswith("[pyarrow]")
    assert hasattr(df["a"].dtype, "numpy_dtype")
    assert df["a"].dtype.numpy_dtype == "int32"
    
    # Test that SimpleImputer preserves int32 dtype
    imputer = SimpleImputer(strategy=strategy, fill_value=0 if strategy == "constant" else None)
    imputer.fit(df)
    
    # Check that _fit_dtype is correctly set to int32
    assert imputer._fit_dtype == np.dtype("int32")
    
    # Verify that the imputer can work with numpy int32 arrays in transform
    numpy_array = np.array([[5]], dtype=np.int32)
    result = imputer.transform(numpy_array)
    assert result.dtype == np.dtype("int32")
// itkRollingMetrics.h
#ifndef itkRollingMetrics_h
#define itkRollingMetrics_h

namespace itk
{

/**
 * @class RollingAverageCircular
 * @brief ITK implementation of the circular buffer rolling average.
 * @tparam TValue The type of the data points (e.g., float, double, int).
 * @tparam VWindowSize The fixed size of the rolling window.
 */
template <unsigned int N>
class RollingCircularBuffer
{
public:
  // Ensure the window size is positive at compile time
  static_assert(N > 0, "Window size (N) must be greater than 0.");

  RollingCircularBuffer()
    : window_index_(0)
    , current_sum_(0.0)
    , count_(0)
  {}

  /**
   * @brief Adds a new data point and updates the running average.
   * @param newValue The new data point to incorporate.
   * @return The current rolling average.
   */
  double
  update(double newValue)
  {
    // 1. Get the value being replaced (the oldest value).
    double oldValue = window_data_[window_index_];

    // 2. Update the running sum
    current_sum_ -= oldValue;
    current_sum_ += newValue;

    // 3. Replace the old value in the array with the new value.
    window_data_[window_index_] = newValue;

    // 4. Move the index to the next position using the modulo operator.
    window_index_ = (window_index_ + 1) % N;

    // 5. Update the count (only necessary until the buffer is full).
    if (count_ < N)
    {
      count_++;
    }
  }

  double
  getAverage() const
  {
    return current_sum_ / static_cast<double>(count_);
  }

  double
  getMax() const
  {
    double max_value = window_data_[0];
    for (unsigned int i = 1; i < count_; ++i)
    {
      if (window_data_[i] > max_value)
      {
        max_value = window_data_[i];
      }
    }
    return max_value;
  }

private:
  // The size of the array is fixed by the template parameter N.
  std::array<double, N> window_data_ = {};
  unsigned int          window_index_;
  double                current_sum_;
  unsigned int          count_;
};

} // end namespace itk

#endif

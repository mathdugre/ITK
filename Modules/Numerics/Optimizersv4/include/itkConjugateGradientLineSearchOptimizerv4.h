/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkConjugateGradientLineSearchOptimizerv4_h
#define itkConjugateGradientLineSearchOptimizerv4_h

#include "itkGradientDescentLineSearchOptimizerv4.h"
#include "itkOptimizerParameterScalesEstimator.h"
#include "itkWindowConvergenceMonitoringFunction.h"
#include "itkRollingMetrics.h"

namespace itk
{
/**
 * \class ConjugateGradientLineSearchOptimizerv4Template
 *  \brief Conjugate gradient descent optimizer with a golden section line search for nonlinear optimization.
 *
 * ConjugateGradientLineSearchOptimizer implements a conjugate gradient descent optimizer
 * that is followed by a line search to find the best value for the learning rate.
 * At each iteration the current position is updated according to
 *
 * \f[
 *        p_{n+1} = p_n
 *                + \mbox{learningRateByGoldenSectionLineSearch}
 *                 \, d
 * \f]
 *
 * where d is defined as the Polak-Ribiere conjugate gradient.
 *
 * Options are identical to the superclass's.
 *
 * \ingroup ITKOptimizersv4
 */
template <typename TInternalComputationValueType>
class ITK_TEMPLATE_EXPORT ConjugateGradientLineSearchOptimizerv4Template
  : public GradientDescentLineSearchOptimizerv4Template<TInternalComputationValueType>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ConjugateGradientLineSearchOptimizerv4Template);

  /** Standard class type aliases. */
  using Self = ConjugateGradientLineSearchOptimizerv4Template;
  using Superclass = GradientDescentLineSearchOptimizerv4Template<TInternalComputationValueType>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** \see LightObject::GetNameOfClass() */
  itkOverrideGetNameOfClassMacro(ConjugateGradientLineSearchOptimizerv4Template);

  /** New macro for creation of through a Smart Pointer */
  itkNewMacro(Self);

  /** It should be possible to derive the internal computation type from the class object. */
  using InternalComputationValueType = TInternalComputationValueType;

  /** Derivative type */
  using typename Superclass::DerivativeType;

  /** Metric type over which this class is templated */
  using typename Superclass::MeasureType;

  /** Type for the convergence checker */
  using ConvergenceMonitoringType = itk::Function::WindowConvergenceMonitoringFunction<TInternalComputationValueType>;

  void
  StartOptimization(bool doOnlyInitialization = false) override;

  unsigned int
  GetVPRECPrecision() const override
  {
    const static std::map<float, unsigned int> precision_map = {
      { 8.0f, 8 }, { 11.0f, 11 }, { 24.0f, 24 }, { 32.0f, 32 }, { std::numeric_limits<float>::max(), 64 }
    };
    // Find the first element whose key (max_estimate_for_range) is
    // NOT less than (i.e., is greater than or equal to) m_PminEstimate.
    auto it = precision_map.lower_bound(m_PminEstimate);
    if (it != precision_map.end())
    {
      return it->second;
    }

    // Fallback, though with the max() entry, this should be unreachable
    return 64;
  }

  TInternalComputationValueType
  GetPminEstimate() const override
  {
    return m_PminEstimate;
  }

  double
  GetRollingAveragePminEstimate() const override
  {
    return m_RollingAveragePminEstimate;
  }

  double
  GetRollingMaxPminEstimate() const override
  {
    return m_RollingMaxPminEstimate;
  }

  /** Get the estimated Lipschitz constant */
  TInternalComputationValueType
  GetLipschitzEstimate() const override
  {
    return m_LipschitzEstimate;
  }
  TInternalComputationValueType
  GetParametersTwoNorm() const override
  {
    return this->m_Metric->GetParameters().two_norm();
  }
  TInternalComputationValueType
  GetGradientTwoNorm() const override
  {
    return this->m_Gradient.two_norm();
  }

protected:
  /** Advance one Step following the gradient direction.
   * Includes transform update. */
  void
  AdvanceOneStep() override;

  /** Default constructor */
  ConjugateGradientLineSearchOptimizerv4Template() = default;

  /** Destructor */
  ~ConjugateGradientLineSearchOptimizerv4Template() override = default;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  DerivativeType                  m_LastGradient{};
  DerivativeType                  m_ConjugateGradient{};
  TInternalComputationValueType   m_LipschitzEstimate = NumericTraits<TInternalComputationValueType>::ZeroValue();
  TInternalComputationValueType   m_PminEstimate = NumericTraits<TInternalComputationValueType>::ZeroValue();
  RollingCircularBuffer<5>        m_RollingAveragePminEstimator;
  double                          m_RollingAveragePminEstimate{ 0.0 };
  double                          m_RollingMaxPminEstimate{ 0.0 };
};

/** This helps to meet backward compatibility */
using ConjugateGradientLineSearchOptimizerv4 = ConjugateGradientLineSearchOptimizerv4Template<double>;

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkConjugateGradientLineSearchOptimizerv4.hxx"
#endif

#endif

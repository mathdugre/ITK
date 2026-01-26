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
#ifndef itkConjugateGradientLineSearchOptimizerv4_hxx
#define itkConjugateGradientLineSearchOptimizerv4_hxx

// #include <interflop/interflop.h>

namespace itk
{

/**
 *PrintSelf
 */
template <typename TInternalComputationValueType>
void
ConjugateGradientLineSearchOptimizerv4Template<TInternalComputationValueType>::PrintSelf(std::ostream & os,
                                                                                         Indent         indent) const
{
  Superclass::PrintSelf(os, indent);
}

template <typename TInternalComputationValueType>
void
ConjugateGradientLineSearchOptimizerv4Template<TInternalComputationValueType>::StartOptimization(
  bool doOnlyInitialization)
{
  this->m_ConjugateGradient.SetSize(this->m_Metric->GetNumberOfParameters());
  this->m_ConjugateGradient.Fill(TInternalComputationValueType{});
  this->m_LastGradient.SetSize(this->m_Metric->GetNumberOfParameters());
  this->m_LastGradient.Fill(TInternalComputationValueType{});
  Superclass::StartOptimization(doOnlyInitialization);
}

/**
 * Advance one Step following the gradient direction
 */
template <typename TInternalComputationValueType>
void
ConjugateGradientLineSearchOptimizerv4Template<TInternalComputationValueType>::AdvanceOneStep()
{
  itkDebugMacro("AdvanceOneStep");

  this->ModifyGradientByScales();
  if (this->m_CurrentIteration == 0)
  {
    this->EstimateLearningRate();
  }

  TInternalComputationValueType gamma{};
  TInternalComputationValueType gammaDenom = inner_product(this->m_LastGradient, this->m_LastGradient);
  if (gammaDenom > itk::NumericTraits<TInternalComputationValueType>::epsilon())
  {
    gamma = inner_product(this->m_Gradient - this->m_LastGradient, this->m_Gradient) / gammaDenom;
  }

  /** Modified Polak-Ribiere restart conditions */
  if (gamma < 0 || gamma > 5)
  {
    gamma = 0;
  }
  this->m_LastGradient = this->m_Gradient;
  this->m_ConjugateGradient = this->m_Gradient + this->m_ConjugateGradient * gamma;
  this->m_Gradient = this->m_ConjugateGradient;

  /* Estimate a learning rate for this step */
  this->m_LineSearchIterations = 0;
  this->m_LearningRate = this->GoldenSectionSearch(
    this->m_LearningRate * this->m_LowerLimit, this->m_LearningRate, this->m_LearningRate * this->m_UpperLimit);

  /* Begin threaded gradient modification of m_Gradient variable. */
  this->ModifyGradientByLearningRate();

  try
  {
    vnl_vector<TInternalComputationValueType> LastParameters = (this->m_Metric->GetParameters());

    /* Pass gradient to transform and let it do its own updating. */
    this->m_Metric->UpdateTransformParameters(this->m_Gradient);

    /* Estimate Lipschitz constant */
    vnl_vector<TInternalComputationValueType> gradDiff = this->m_Gradient - this->m_LastGradient;
    vnl_vector<TInternalComputationValueType> currentParams = (this->m_Metric->GetParameters());
    vnl_vector<TInternalComputationValueType> paramDiff = currentParams - LastParameters;

    TInternalComputationValueType gradNorm = gradDiff.two_norm();
    TInternalComputationValueType paramNorm = paramDiff.two_norm();

    this->m_LipschitzEstimate = gradNorm / paramNorm;

    // --- VPREC AMP ---
    // Compute pmin estimate
    const double CONSTANT_FACTOR = 4.0 + 3.0 * std::sqrt(2.0);
    double       arg =
      (CONSTANT_FACTOR * this->m_LipschitzEstimate * this->GetParametersTwoNorm()) / this->GetGradientTwoNorm();

    // Ensure arg is in log2 domain
    double new_pmin_estimate = (arg <= 0.0) ? -std::numeric_limits<double>::infinity() : std::log2(arg);

    // Store current Pmin estimate and update rolling metrics
    this->m_PminEstimate = new_pmin_estimate;
    this->m_RollingAveragePminEstimator.update(new_pmin_estimate);
    this->m_RollingAveragePminEstimate = this->m_RollingAveragePminEstimator.getAverage();
    this->m_RollingMaxPminEstimate = this->m_RollingAveragePminEstimator.getMax();

    // Assign new VPREC precision
    this->UpdatePrecision(this->m_RollingAveragePminEstimate);
    unsigned int vprec_precision = this->m_VPRECPrecision;
    interflop_call(INTERFLOP_SET_PRECISION_BINARY32, vprec_precision);
    interflop_call(INTERFLOP_SET_PRECISION_BINARY64, vprec_precision);
    interflop_call(INTERFLOP_SET_RANGE_BINARY32, 8);
    interflop_call(INTERFLOP_SET_RANGE_BINARY64, 8);
  }
  catch (const ExceptionObject &)
  {
    this->m_StopCondition = StopConditionObjectToObjectOptimizerEnum::UPDATE_PARAMETERS_ERROR;
    this->m_StopConditionDescription << "UpdateTransformParameters error";
    this->StopOptimization();
    // Pass exception to caller
    throw;
  }

  this->InvokeEvent(IterationEvent());
}

} // namespace itk

#endif

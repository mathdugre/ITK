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
#ifndef LinearInterpolateImageFunction_h
#define LinearInterpolateImageFunction_h

#include "itkInterpolateImageFunction.h"
#include "itkVariableLengthVector.h"
#include <algorithm> // For max.

#include <cstdlib> // For lcg()
#include <ctime>   // For seeding with time

namespace itk
{
/**
 * \class LinearInterpolateImageFunction
 * \brief Linearly interpolate an image at specified positions.
 *
 * LinearInterpolateImageFunction linearly interpolates image intensity at
 * a non-integer pixel position. This class is templated
 * over the input image type and the coordinate representation type
 * (e.g. float or double).
 *
 * This function works for N-dimensional images.
 *
 * This function works for images with scalar and vector pixel
 * types, and for images of type VectorImage.
 *
 * \sa VectorLinearInterpolateImageFunction
 *
 * \ingroup ImageFunctions ImageInterpolators
 * \ingroup ITKImageFunction
 *
 * \sphinx
 * \sphinxexample{Core/ImageFunction/LinearlyInterpolatePositionInImage,Linearly Interpolate Position In Image}
 * \endsphinx
 */
template <typename TInputImage, typename TCoordRep = double, typename TRealType = typename NumericTraits<typename TInputImage::PixelType>::RealType>
class ITK_TEMPLATE_EXPORT LinearInterpolateImageFunction : public InterpolateImageFunction<TInputImage, TCoordRep, TRealType>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(LinearInterpolateImageFunction);

  /** Standard class type aliases. */
  using Self = LinearInterpolateImageFunction;
  using Superclass = InterpolateImageFunction<TInputImage, TCoordRep, TRealType>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** \see LightObject::GetNameOfClass() */
  itkOverrideGetNameOfClassMacro(LinearInterpolateImageFunction);

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** OutputType type alias support */
  using typename Superclass::OutputType;

  /** InputImageType type alias support */
  using typename Superclass::InputImageType;

  /** InputPixelType type alias support */
  using typename Superclass::InputPixelType;

  /** RealType type alias support */
  using typename Superclass::RealType;

  /** Dimension underlying input image. */
  static constexpr unsigned int ImageDimension = Superclass::ImageDimension;

  /** Index type alias support */
  using typename Superclass::IndexType;

  /** Size type alias support */
  using typename Superclass::SizeType;

  /** ContinuousIndex type alias support */
  using typename Superclass::ContinuousIndexType;
  using InternalComputationType = typename ContinuousIndexType::ValueType;

  /** Evaluate the function at a ContinuousIndex position
   *
   * Returns the linearly interpolated image intensity at a
   * specified point position. No bounds checking is done.
   * The point is assume to lie within the image buffer.
   *
   * ImageFunction::IsInsideBuffer() can be used to check bounds before
   * calling the method. */
  OutputType
  EvaluateAtContinuousIndex(const ContinuousIndexType & index) const override
  {
    return this->EvaluateOptimized(Dispatch<ImageDimension>(), index);
  }

  SizeType
  GetRadius() const override
  {
    return SizeType::Filled(1);
  }

protected:
  LinearInterpolateImageFunction() = default;
  ~LinearInterpolateImageFunction() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

private:
  // LCG parameters
  const static unsigned long a = 1664525;
  const static unsigned long c = 1013904223;
  const static unsigned long m = 4294967296;                                   // 2^32
  inline static unsigned long       seed = static_cast<unsigned long>(time(nullptr)); // Seed value

  struct DispatchBase
  {};
  template <unsigned int>
  struct Dispatch : public DispatchBase
  {};

  // LCG function
  inline unsigned long
  lcg() const
  {
    seed = (a * seed + c) % m;
    return seed;
  }

  inline OutputType
  EvaluateOptimized(const Dispatch<0> &, const ContinuousIndexType &) const
  {
    return 0;
  }

  inline OutputType
  EvaluateOptimized(const Dispatch<1> &, const ContinuousIndexType & index) const
  {
    IndexType basei;

    basei[0] = std::max(Math::Floor<IndexValueType>(index[0]), this->m_StartIndex[0]);
    const InternalComputationType distance0 = index[0] - static_cast<InternalComputationType>(basei[0]);

    const TInputImage * const inputImagePtr = this->GetInputImage();

    // Select a random point weighted by their distance
    InternalComputationType randomValue = static_cast<InternalComputationType>(this->lcg());
    if (randomValue < distance0)
    {
      ++basei[0];
      if (basei[0] > this->m_EndIndex[0])
      {
        --basei[0];
      }
    }

    return static_cast<OutputType>(inputImagePtr->GetPixel(basei));
  }

  inline OutputType
  EvaluateOptimized(const Dispatch<2> &, const ContinuousIndexType & index) const
  {
    IndexType basei;

    basei[0] = std::max(Math::Floor<IndexValueType>(index[0]), this->m_StartIndex[0]);
    const InternalComputationType distance0 = index[0] - static_cast<InternalComputationType>(basei[0]);

    basei[1] = std::max(Math::Floor<IndexValueType>(index[1]), this->m_StartIndex[1]);
    const InternalComputationType distance1 = index[1] - static_cast<InternalComputationType>(basei[1]);

    const TInputImage * const inputImagePtr = this->GetInputImage();

    // Select a random point weighted by their distance
    const InternalComputationType randomValue = static_cast<InternalComputationType>(this->lcg());
    if (randomValue < distance0)
    {
      ++basei[0];
      if (basei[0] > this->m_EndIndex[0])
      {
        --basei[0];
      }
    }
    if (randomValue < distance1)
    {
      ++basei[1];
      if (basei[1] > this->m_EndIndex[1])
      {
        --basei[1];
      }
    }

    return static_cast<OutputType>(inputImagePtr->GetPixel(basei));
  }

  inline OutputType
  EvaluateOptimized(const Dispatch<3> &, const ContinuousIndexType & index) const
  {
    IndexType basei;

    basei[0] = std::max(Math::Floor<IndexValueType>(index[0]), this->m_StartIndex[0]);
    const InternalComputationType distance0 = index[0] - static_cast<InternalComputationType>(basei[0]);

    basei[1] = std::max(Math::Floor<IndexValueType>(index[1]), this->m_StartIndex[1]);
    const InternalComputationType distance1 = index[1] - static_cast<InternalComputationType>(basei[1]);

    basei[2] = std::max(Math::Floor<IndexValueType>(index[2]), this->m_StartIndex[2]);
    const InternalComputationType distance2 = index[2] - static_cast<InternalComputationType>(basei[2]);

    const TInputImage * const inputImagePtr = this->GetInputImage();

    // Select a random point weighted by their distance
    const InternalComputationType randomValue = static_cast<InternalComputationType>(this->lcg());
    if (randomValue < distance0)
    {
      ++basei[0];
      if (basei[0] > this->m_EndIndex[0])
      {
        --basei[0];
      }
    }
    if (randomValue < distance1)
    {
      ++basei[1];
      if (basei[1] > this->m_EndIndex[1])
      {
        --basei[1];
      }
    }
    if (randomValue < distance2)
    {
      ++basei[2];
      if (basei[2] > this->m_EndIndex[2])
      {
        --basei[2];
      }
    }

    return static_cast<OutputType>(inputImagePtr->GetPixel(basei));
  }

  inline OutputType
  EvaluateOptimized(const DispatchBase &, const ContinuousIndexType & index) const
  {
    return this->EvaluateUnoptimized(index);
  }

  /** Evaluate interpolator at image index position. */
  virtual inline OutputType
  EvaluateUnoptimized(const ContinuousIndexType & index) const;

  /** \brief A method to generically set all components to zero
   */
  template <typename RealTypeScalarRealType>
  void
  MakeZeroInitializer(const TInputImage * const                      inputImagePtr,
                      VariableLengthVector<RealTypeScalarRealType> & tempZeros) const
  {
    // Variable length vector version to get the size of the pixel correct.
    constexpr typename TInputImage::IndexType idx = { { 0 } };
    const typename TInputImage::PixelType &   tempPixel = inputImagePtr->GetPixel(idx);
    const unsigned int                        sizeOfVarLengthVector = tempPixel.GetSize();
    tempZeros.SetSize(sizeOfVarLengthVector);
    tempZeros.Fill(RealTypeScalarRealType{});
  }

  template <typename RealTypeScalarRealType>
  void
  MakeZeroInitializer(const TInputImage * const itkNotUsed(inputImagePtr), RealTypeScalarRealType & tempZeros) const
  {
    // All other cases
    tempZeros = RealTypeScalarRealType{};
  }
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkLinearInterpolateImageFunction.hxx"
#endif

#endif

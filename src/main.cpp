/*=============================================================================

MIT License

Copyright (c) 2018 Ville Ruusutie

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

=============================================================================*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
//----------------------------------------------------------------------------

// Transfer functions and their derivatives
inline float sigmoid( float fValue )			{ return 1.0f / ( 1.0f + expf( -fValue ) ); }
inline float sigmoidDerivative( float fValue )	{ return fValue * ( 1.0f - fValue ); }
inline float relu( float fValue )				{ return fmax( 0.0f, fValue ); }
inline float reluDerivative( float fValue )		{ return fValue > 0.0f ? 1.0f : 0.0f; }
inline float softplus( float fValue )			{ return logf( 1.0f + expf( fValue )  ); }
inline float softplusDerivative( float fValue )	{ return sigmoid( fValue ); }
inline float elu( float fValue )				{ return fValue >= 0.0f ? fValue : ( expf( fValue ) - 1.0f ); }
inline float eluDerivative( float fValue )		{ return fValue >= 0.0f ? 1.0f : expf( fValue ); }

inline float transfer( float fValue )			{ return elu( fValue ); }
inline float transferDerivative( float fValue )	{ return eluDerivative( fValue ); }
//----------------------------------------------------------------------------

inline float randomFloat()
{
	return ( float( rand() ) / float( RAND_MAX ) ) * 0.4f + 0.5f;
}
//----------------------------------------------------------------------------

void randomize( float* pValues, size_t count )
{
	for( size_t i = 0; i < count; ++i )
	{
		pValues[ i ] = randomFloat();
	}
}
//----------------------------------------------------------------------------

void print( const char* strName, const float* pValues, size_t count )
{
	for( size_t i = 0; i < count; ++i )
	{
		printf( "%s[%d] %.5f\n", strName, ( int )i, pValues[ i ] );
	}
}
//----------------------------------------------------------------------------

struct Layer
{
	Layer( size_t inputCount, size_t outputCount )
		: m_inputCount( inputCount )
		, m_outputCount( outputCount )
	{
		m_pWeights = new float[ inputCount * outputCount ];
		m_pBiases = new float[ outputCount ];

		randomize( m_pWeights, inputCount * outputCount );
		randomize( m_pBiases, outputCount );
	}
	//------------------------------------------------------------------------

	~Layer()
	{
		delete [] m_pWeights;
		delete [] m_pBiases;
	}
	//------------------------------------------------------------------------

	void propagate( const float* pInputs, float* pOutputs ) const
	{
		for( size_t o = 0; o < m_outputCount; ++o )
		{
			const float* pWeights = &m_pWeights[ m_inputCount * o ];
			float fActivation = m_pBiases[ o ];
			for( size_t i = 0; i < m_inputCount; ++i )
			{
				fActivation += pInputs[ i ] * pWeights[ i ];
			}
			pOutputs[ o ] = transfer( fActivation );
		}
	}
	//------------------------------------------------------------------------

	void updateWeights( const float* pInputs, const float* pDeltas, float fLearningRate )
	{
		for( size_t o = 0; o < m_outputCount; ++o )
		{
			float* pWeights = &m_pWeights[ m_inputCount * o ];
			const float fChange = pDeltas[ o ] * fLearningRate;

			for( size_t i = 0; i < m_inputCount; ++i )
			{
				pWeights[ i ] += fChange * pInputs[ i ];
			}
			m_pBiases[ o ] += fChange;
		}
	}
	//------------------------------------------------------------------------
	
	float computeOutputDeltas( const float* pOutputValues, const float* pExpectedValues, float* pDeltas ) const
	{
		float fTotalQuadraticError = 0.0f;
		for( size_t o = 0; o < m_outputCount; ++o )
		{
			const float fOutput = pOutputValues[ o ];
			const float fError = pExpectedValues[ o ] - fOutput;
			pDeltas[ o ] = fError * transferDerivative( fOutput );
			fTotalQuadraticError += fError * fError;
		}
		return fTotalQuadraticError;
	}
	//------------------------------------------------------------------------

	void computeDeltas( const Layer* pNextLayer, const float* pNextDeltas, const float* pValues, float* pDeltas ) const
	{
		for( size_t o = 0; o < m_outputCount; ++o )
		{
			float fError = 0.0f;
			for( size_t i = 0; i < pNextLayer->m_outputCount; ++i )
			{
				fError += pNextDeltas[ i ] * pNextLayer->m_pWeights[ m_outputCount * i + o ];
			}

			pDeltas[ o ] = fError * transferDerivative( pValues[ o ] );
		}
	}
	//------------------------------------------------------------------------

	size_t getInputCount() const	{ return m_inputCount; }
	size_t getOutputCount() const	{ return m_outputCount; }
	//------------------------------------------------------------------------

private:
	size_t	m_inputCount;
	size_t	m_outputCount;
	float*	m_pWeights;
	float*	m_pBiases;
};
//----------------------------------------------------------------------------

struct NeuralNet
{
	NeuralNet( size_t inputCount, size_t hiddenCount, size_t outputCount )
		: m_hiddenLayer( inputCount, hiddenCount )
		, m_outputLayer( hiddenCount, outputCount )
	{
	}
	//------------------------------------------------------------------------

	void evaluate( const float* pInputs, float* pOutputs ) const
	{
		float* pHiddenOutputs = ( float* )alloca( m_hiddenLayer.getOutputCount() * sizeof( float ) );
		m_hiddenLayer.propagate( pInputs, pHiddenOutputs );
		m_outputLayer.propagate( pHiddenOutputs, pOutputs );
	}
	//------------------------------------------------------------------------

	void train( const float* pAllInputs, const float* pAllExpectedOutputs, size_t testCount, size_t epochCount, float fLearningRate )
	{
		const size_t inputCount = m_hiddenLayer.getInputCount();
		const size_t hiddenCount = m_hiddenLayer.getOutputCount();
		const size_t outputCount = m_outputLayer.getOutputCount();

		float* pHiddenValues = ( float* )alloca( hiddenCount * sizeof( float ) );
		float* pHiddenDeltas = ( float* )alloca( hiddenCount * sizeof( float ) );
		float* pOutputValues = ( float* )alloca( outputCount * sizeof( float ) );
		float* pOutputDeltas = ( float* )alloca( outputCount * sizeof( float ) );

		for( int epoch = 0; epoch < epochCount; ++epoch )
		{
			float fTotalQuadraticError = 0.0f;

			for( size_t test = 0; test < testCount; ++test )
			{
				const float* pInputs = &pAllInputs[ test * inputCount ];
				const float* pExpectedOutputs = &pAllExpectedOutputs[ test * outputCount ];

				// Propagate to get current state
				m_hiddenLayer.propagate( pInputs, pHiddenValues );
				m_outputLayer.propagate( pHiddenValues, pOutputValues );

				// Backpropagate errors to deltas
				fTotalQuadraticError += m_outputLayer.computeOutputDeltas( pOutputValues, pExpectedOutputs, pOutputDeltas );
				m_hiddenLayer.computeDeltas( &m_outputLayer, pOutputDeltas, pHiddenValues, pHiddenDeltas );

				// Update weights and biases with deltas
				m_outputLayer.updateWeights( pHiddenValues, pOutputDeltas, fLearningRate );
				m_hiddenLayer.updateWeights( pInputs, pHiddenDeltas, fLearningRate );
			}

			printf( "epoch: %d  error: %.3f\n", epoch, fTotalQuadraticError );
		}
	}
	//------------------------------------------------------------------------

private:
	Layer m_hiddenLayer;
	Layer m_outputLayer;
};
//----------------------------------------------------------------------------

constexpr size_t s_testCount = 4;

static float s_testInputData[] = {
	0.0f,
	0.2f,
	0.8f,
	1.0f,
};

static float s_testOutputData[] = {
	1.0f,
	0.8f,
	0.2f,
	0.0f,
};
//----------------------------------------------------------------------------

int main( int, const char** )
{
	NeuralNet net( 1, 8, 1 );

	// Learn
	const size_t epochCount = 50;
	const float fLearningRate = 0.2f;
	net.train( s_testInputData, s_testOutputData, s_testCount, epochCount, fLearningRate );

	// Check if learned
	float outputs[ 1 ] = {0};
	for( size_t i = 0; i < s_testCount; ++i )
	{
		net.evaluate( &s_testInputData[ i ], outputs );
		printf( "input %.3f  outputs %.3f\n", s_testInputData[ i ], outputs[ 0 ] );
	}

	return 0;
}

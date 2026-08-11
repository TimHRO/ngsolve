#ifndef FILE_IFGFOPERATOR
#define FILE_IFGFOPERATOR

#include "fmmoperator.hpp"
#include <Eigen/Dense>
#include <ifgf/ifgf_library.hpp>
#include <ngbem.hpp>

#define USE_IFGF
namespace ngsbem
{

    template <typename KERNEL>
    class IFGF_Operator : public FMM_Operator<KERNEL>
    {
    public:
        IFGF_Operator(KERNEL _kernel, Array<Vec<3>> _xpts, Array<Vec<3>> _ypts,
                      Array<Vec<3>> _xnv, Array<Vec<3>> _ynv, const FMM_Parameters &fmm_params)
            : FMM_Operator<KERNEL>(_kernel, std::move(_xpts), std::move(_ypts), std::move(_xnv), std::move(_ynv), fmm_params)
        {
        }
    };
#ifdef USE_IFGF

    // -----------------------------------------------------------------------
    //  Single layer Helmholtz
    // -----------------------------------------------------------------------
    template <>
    class IFGF_Operator<HelmholtzSLKernel<3, 1, Complex>> : public Base_FMM_Operator<std::complex<double>>
    {
        typedef HelmholtzSLKernel<3, 1, Complex> KERNEL;
        typedef ifgf::HelmholtzSL3D OperatorType;
        typedef Base_FMM_Operator<std::complex<double>> BASE;

    IFGF_Parameters ifgf_params;

    protected:
        std::unique_ptr<OperatorType> op;
        KERNEL kernel;

    public:
        IFGF_Operator(KERNEL _kernel, Array<Vec<3>> _xpts, Array<Vec<3>> _ypts,
                      Array<Vec<3>> _xnv, Array<Vec<3>> _ynv, const IntOp_Parameters &io_params)
            : BASE(std::move(_xpts), std::move(_ypts), std::move(_xnv), std::move(_ynv),
                   KERNEL::Shape(), io_params),
              kernel(_kernel)
        {

	    ifgf_params = static_cast<IFGF_Parameters>(io_params);
            std::cout << "creating ifgf helmholtz SL op" << std::endl;
            //std::complex<float> waveNumber = static_cast<std::complex<float>>(kernel.GetKappa());
            std::complex<double> waveNumber = kernel.GetKappa();
            std::cout << waveNumber << std::endl;

            size_t leafSize = ifgf_params.maxLeafSize;
            size_t order = ifgf_params.order;
            int n_elem = ifgf_params.n_elements;
            double tol = ifgf_params.tolerance;
            double maxk = ifgf_params.maxk;
            double minSigma = ifgf_params.minSigma;

            
            std::cout << "order passed to ngsolve " << ifgf_params.order << std::endl;
	    std::cout << "maxk passed to ngsolve " << maxk << std::endl;
            std::cout << "minSigma passed to ngsolve " << minSigma << std::endl;

            std::vector<double> src_buf(3 * xpts.Size());
            std::vector<double> tgt_buf(3 * ypts.Size());
            for (int i = 0; i < xpts.Size(); i++) {
                src_buf[3*i+0] = xpts[i][0];
                src_buf[3*i+1] = xpts[i][1];
                src_buf[3*i+2] = xpts[i][2];
            }
            for (int i = 0; i < ypts.Size(); i++) {
                tgt_buf[3*i+0] = ypts[i][0];
                tgt_buf[3*i+1] = ypts[i][1];
                tgt_buf[3*i+2] = ypts[i][2];
            }

            op = make_unique<OperatorType>(waveNumber, leafSize, order, n_elem, tol, maxk, minSigma);
            op->init(src_buf.data(), xpts.Size(), tgt_buf.data(), ypts.Size());
        }

        void Mult(const BaseVector &x, BaseVector &y) const override
        {
            std::cout << "in mult ngsolve" << std::endl;
            static Timer tall("ngbem ifgf apply HelmholtzSL");
            RegionTimer reg(tall);

            auto fx = x.FV<Complex>();
            auto fy = y.FV<Complex>();

            // cast to float, no cast inside library
            Eigen::Vector<std::complex<float>, Eigen::Dynamic> weights_f =
                Eigen::Map<const Eigen::Vector<std::complex<double>, Eigen::Dynamic>>(
                    reinterpret_cast<const std::complex<double>*>(fx.Data()), fx.Size())
                .cast<std::complex<float>>();

            Eigen::Vector<std::complex<float>, Eigen::Dynamic> result_f(fy.Size());

            op->mult(weights_f.data(), weights_f.size(), result_f.data(), result_f.size());

            Eigen::Map<Eigen::Vector<std::complex<double>, Eigen::Dynamic>>(
                reinterpret_cast<std::complex<double>*>(fy.Data()), fy.Size())
            = result_f.cast<std::complex<double>>();
        }

        BaseMatrix::OperatorInfo GetOperatorInfo() const override
        {
            return { "IFGF_Operator HelmholtzSL", this->Height(), this->Width() };
        }

        FMMOperatorInfo GetFMMInfo() const override
        {
            FMMOperatorInfo info;
            info.kernel_name = KERNEL::Name();
            info.source_size = xpts.Size();
            info.target_size = ypts.Size();
            info.kappa = kernel.GetKappa();
            info.parameters = fmm_params;
            return info;
        }
    };

    // -----------------------------------------------------------------------
    //  Double layer Helmholtz
    // -----------------------------------------------------------------------
    template <>
    class IFGF_Operator<HelmholtzDLKernel<3, 1, Complex>> : public Base_FMM_Operator<std::complex<double>>
    {
        typedef HelmholtzDLKernel<3, 1, Complex> KERNEL;
        typedef ifgf::HelmholtzDL3D OperatorType;
        typedef Base_FMM_Operator<std::complex<double>> BASE;
    
    IFGF_Parameters ifgf_params;

    protected:
        std::unique_ptr<OperatorType> op;
        KERNEL kernel;

    public:
        IFGF_Operator(KERNEL _kernel, Array<Vec<3>> _xpts, Array<Vec<3>> _ypts,
                      Array<Vec<3>> _xnv, Array<Vec<3>> _ynv, const IntOp_Parameters &io_params)
            : BASE(std::move(_xpts), std::move(_ypts), std::move(_xnv), std::move(_ynv),
                   KERNEL::Shape(), io_params),
              kernel(_kernel)
        {

            ifgf_params = static_cast<IFGF_Parameters>(io_params);
            std::cout << "creating ifgf helmholtz DL op" << std::endl;
            //std::complex<float> waveNumber = static_cast<std::complex<float>>(kernel.GetKappa());
            std::complex<double> waveNumber = kernel.GetKappa();
            std::cout << waveNumber << std::endl;

            size_t leafSize = ifgf_params.maxLeafSize;
            size_t order = ifgf_params.order;
            int n_elem = ifgf_params.n_elements;
            double tol = ifgf_params.tolerance;
            double maxk = ifgf_params.maxk;
            double minSigma = ifgf_params.minSigma;
            
            std::cout << "order passed to ngsolve " << ifgf_params.order << std::endl;
            std::cout << "maxk passed to ngsolve " << maxk << std::endl;
            std::cout << "minSigma passed to ngsolve " << minSigma << std::endl;


            std::vector<double> src_buf(3 * xpts.Size());
            std::vector<double> tgt_buf(3 * ypts.Size());
            std::vector<double> nrm_buf(3 * xnv.Size());
            for (int i = 0; i < xpts.Size(); i++) {
                src_buf[3*i+0] = xpts[i][0];
                src_buf[3*i+1] = xpts[i][1];
                src_buf[3*i+2] = xpts[i][2];
            }
            for (int i = 0; i < ypts.Size(); i++) {
                tgt_buf[3*i+0] = ypts[i][0];
                tgt_buf[3*i+1] = ypts[i][1];
                tgt_buf[3*i+2] = ypts[i][2];
            }
            for (int i = 0; i < xnv.Size(); i++) {
                nrm_buf[3*i+0] = xnv[i][0];
                nrm_buf[3*i+1] = xnv[i][1];
                nrm_buf[3*i+2] = xnv[i][2];
            }

            op = make_unique<OperatorType>(waveNumber, leafSize, order, n_elem, tol, maxk, minSigma);
            op->init(src_buf.data(), xpts.Size(), tgt_buf.data(), ypts.Size(),
                     nrm_buf.data(), xnv.Size());
        }

        void Mult(const BaseVector &x, BaseVector &y) const override
        {
            static Timer tall("ngbem ifgf apply HelmholtzDL");
            RegionTimer reg(tall);

            auto fx = x.FV<Complex>();
            auto fy = y.FV<Complex>();

            Eigen::Vector<std::complex<float>, Eigen::Dynamic> weights_f =
                Eigen::Map<const Eigen::Vector<std::complex<double>, Eigen::Dynamic>>(
                    reinterpret_cast<const std::complex<double>*>(fx.Data()), fx.Size())
                .cast<std::complex<float>>();

            Eigen::Vector<std::complex<float>, Eigen::Dynamic> result_f(fy.Size());

            op->mult(weights_f.data(), weights_f.size(), result_f.data(), result_f.size());

            Eigen::Map<Eigen::Vector<std::complex<double>, Eigen::Dynamic>>(
                reinterpret_cast<std::complex<double>*>(fy.Data()), fy.Size())
            = result_f.cast<std::complex<double>>();
        }

        BaseMatrix::OperatorInfo GetOperatorInfo() const override
        {
            return { "IFGF_Operator HelmholtzDL", this->Height(), this->Width() };
        }

        FMMOperatorInfo GetFMMInfo() const override
        {
            FMMOperatorInfo info;
            info.kernel_name = KERNEL::Name();
            info.source_size = xpts.Size();
            info.target_size = ypts.Size();
            info.kappa = kernel.GetKappa();
            info.parameters = fmm_params;
            return info;
        }
    };

    // -----------------------------------------------------------------------
    //  Combined field Helmholtz
    // -----------------------------------------------------------------------
    template <>
    class IFGF_Operator<CombinedFieldKernel<3, 1, Complex>> : public Base_FMM_Operator<std::complex<double>>
    {
        typedef CombinedFieldKernel<3, 1, Complex> KERNEL;
        typedef ifgf::HelmholtzCF3D OperatorType;
        typedef Base_FMM_Operator<std::complex<double>> BASE;

    IFGF_Parameters ifgf_params;
    protected:
        std::unique_ptr<OperatorType> op;
        KERNEL kernel;

    public:
        IFGF_Operator(KERNEL _kernel, Array<Vec<3>> _xpts, Array<Vec<3>> _ypts,
                      Array<Vec<3>> _xnv, Array<Vec<3>> _ynv, const IntOp_Parameters &io_params)
            : BASE(std::move(_xpts), std::move(_ypts), std::move(_xnv), std::move(_ynv),
                   KERNEL::Shape(), io_params),
              kernel(_kernel)
        {

            ifgf_params = static_cast<IFGF_Parameters>(io_params);
            std::cout << "creating ifgf helmholtz CF op" << std::endl;
            //std::complex<float> waveNumber = static_cast<std::complex<float>>(kernel.GetKappa());
            std::complex<double> waveNumber = kernel.GetKappa();
            std::cout << waveNumber << std::endl;

            size_t leafSize = ifgf_params.maxLeafSize;
            size_t order = ifgf_params.order;
            int n_elem = ifgf_params.n_elements;
            double tol = ifgf_params.tolerance;
            double maxk = ifgf_params.maxk;
            double minSigma = ifgf_params.minSigma;
            
	    std::cout << "order passed to ngsolve " << ifgf_params.order << std::endl;
            std::cout << "maxk passed to ngsolve " << maxk << std::endl;
            std::cout << "minSigma passed to ngsolve " << minSigma << std::endl;


            std::vector<double> src_buf(3 * xpts.Size());
            std::vector<double> tgt_buf(3 * ypts.Size());
            std::vector<double> nrm_buf(3 * xnv.Size());
            for (int i = 0; i < xpts.Size(); i++) {
                src_buf[3*i+0] = xpts[i][0];
                src_buf[3*i+1] = xpts[i][1];
                src_buf[3*i+2] = xpts[i][2];
            }
            for (int i = 0; i < ypts.Size(); i++) {
                tgt_buf[3*i+0] = ypts[i][0];
                tgt_buf[3*i+1] = ypts[i][1];
                tgt_buf[3*i+2] = ypts[i][2];
            }
            for (int i = 0; i < xnv.Size(); i++) {
                nrm_buf[3*i+0] = xnv[i][0];
                nrm_buf[3*i+1] = xnv[i][1];
                nrm_buf[3*i+2] = xnv[i][2];
            }

            op = make_unique<OperatorType>(waveNumber, leafSize, order, n_elem, tol, maxk, minSigma);
            op->init(src_buf.data(), xpts.Size(), tgt_buf.data(), ypts.Size(),
                     nrm_buf.data(), xnv.Size());
        }

        void Mult(const BaseVector &x, BaseVector &y) const override
        {
            static Timer tall("ngbem ifgf apply CombinedField");
            RegionTimer reg(tall);

            auto fx = x.FV<Complex>();
            auto fy = y.FV<Complex>();

            Eigen::Vector<std::complex<float>, Eigen::Dynamic> weights_f =
                Eigen::Map<const Eigen::Vector<std::complex<double>, Eigen::Dynamic>>(
                    reinterpret_cast<const std::complex<double>*>(fx.Data()), fx.Size())
                .cast<std::complex<float>>();

            Eigen::Vector<std::complex<float>, Eigen::Dynamic> result_f(fy.Size());

            op->mult(weights_f.data(), weights_f.size(), result_f.data(), result_f.size());

            Eigen::Map<Eigen::Vector<std::complex<double>, Eigen::Dynamic>>(
                reinterpret_cast<std::complex<double>*>(fy.Data()), fy.Size())
            = result_f.cast<std::complex<double>>();
        }

        BaseMatrix::OperatorInfo GetOperatorInfo() const override
        {
            return { "IFGF_Operator CombinedField", this->Height(), this->Width() };
        }

        FMMOperatorInfo GetFMMInfo() const override
        {
            FMMOperatorInfo info;
            info.kernel_name = KERNEL::Name();
            info.source_size = xpts.Size();
            info.target_size = ypts.Size();
            info.kappa = kernel.GetKappa();
            info.parameters = fmm_params;
            return info;
        }
    };

#endif

}

#endif

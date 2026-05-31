#ifndef FILE_IFGFOPERATOR
#define FILE_IFGFOPERATOR

#include "fmmoperator.hpp"
//#include <helmholtz_ifgf.hpp>
//#include <modified_helmholtz_ifgf.hpp>
//#include <double_layer_helmholtz_ifgf.hpp>
//#include <combined_field_helmholtz_ifgf.hpp>
//#include <laplace_ifgf.hpp>
#include <Eigen/Dense>
#include <ifgf_library.hpp>
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

template <>
class IFGF_Operator<HelmholtzSLKernel<3,1,Complex>> : public Base_FMM_Operator<std::complex<double>>
{
    typedef HelmholtzSLKernel<3,1,Complex> KERNEL;
    typedef ModifiedHelmholtzIfgfOperator3d OperatorType;
    typedef Base_FMM_Operator<std::complex<double>> BASE;

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
            std::cout << "creating ifgf helmholtz SL op" << std::endl;
            std::complex<float> waveNumber = static_cast<std::complex<float>>(-1i * kernel.GetKappa());
            std::cout << waveNumber << std::endl;

            size_t leafSize = 300;
            size_t order = 8;
            int n_elem = 1;
            double tol = -1;

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

            op = make_unique<ModifiedHelmholtzIfgfOperator3d>(waveNumber, leafSize, order, n_elem, tol);
            op->init(src_buf.data(), xpts.Size(), tgt_buf.data(), ypts.Size());
        }
	/*
        void Mult(const BaseVector &x, BaseVector &y) const override
        {
            std::cout << "ifgf mult SL" << std::endl;
            static Timer tall("ngbem ifgf apply HelmholtzSL");
            RegionTimer reg(tall);
            auto fx = x.FV<Complex>();
            auto fy = y.FV<Complex>();
            //fy = Complex(0);
	    std::cout << "fx.Size()=" << fx.Size() 
              << " n_src=" << xpts.Size() 
              << " n_tgt=" << ypts.Size()
              << " fy.Size()=" << fy.Size() << std::endl;
            op->mult(fx.Data(), fx.Size(), fy.Data(), fy.Size());
        }*/

	void Mult(const BaseVector &x, BaseVector &y) const override
{
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
    /*
    template <>
    class IFGF_Operator<HelmholtzDLKernel<3>> : public Base_FMM_Operator<std::complex<double>>
    {
        typedef HelmholtzDLKernel<3> KERNEL;
        typedef DoubleLayerHelmholtzIfgfOperator<3> OperatorType;
        typedef Base_FMM_Operator<std::complex<double>> BASE;

    protected:
        std::unique_ptr<OperatorType> op;
        KERNEL kernel;

    public:
        IFGF_Operator(KERNEL _kernel, Array<Vec<3>> _xpts, Array<Vec<3>> _ypts,
                      Array<Vec<3>> _xnv, Array<Vec<3>> _ynv, const FMM_Parameters &fmm_params)
            : BASE(std::move(_xpts), std::move(_ypts), std::move(_xnv), std::move(_ynv), IVec<2>(), fmm_params),
              kernel(_kernel)
        {
            std::cout << "creating ifgf double layer helmholtz op" << std::endl;
            if constexpr (std::is_same<KERNEL, class HelmholtzDLKernel<3>>())
            {
                std::complex<double> waveNumber = 1i * _kernel.GetKappa();
                size_t leafSize = 250;
                size_t order = 8;
                int n_elem = 1;
                double tol = -1;

                std::cout << "size=" << xpts.Size() << std::endl;
                std::cout << "size=" << ypts.Size() << std::endl;

                op = make_unique<DoubleLayerHelmholtzIfgfOperator<3>>(waveNumber, leafSize, order, n_elem, tol);

                auto srcs = Eigen::Map<typename OperatorType::PointArray>(xpts[0].Data(), 3, xpts.Size());
                auto targets = Eigen::Map<typename OperatorType::PointArray>(ypts[0].Data(), 3, ypts.Size());

                auto src_normals = Eigen::Map<typename OperatorType::PointArray>(xnv[0].Data(), 3, xnv.Size());

                op->init(srcs, targets, src_normals);
            }
        }

        void Mult(const BaseVector &x, BaseVector &y) const
        {
            std::cout << "ifgf mult" << std::endl;
            static Timer tall("ngbem fmm apply Helmholtz Double Layer (IFGF)");
            RegionTimer reg(tall);
            auto fx = x.FV<Complex>();
            auto fy = y.FV<Complex>();

            // fy = 0;

            auto weights = Eigen::Map<Eigen::Vector<std::complex<double>, Eigen::Dynamic>>(fx.Data(), fx.Size());
            auto results = op->mult(weights);

            auto y_map = Eigen::Map<Eigen::Vector<std::complex<double>, Eigen::Dynamic>>(fy.Data(), fy.Size());
            y_map = results;
            // y *= 1.0 / (4*M_PI);
        }
    };

    template<>
    class IFGF_Operator<CombinedFieldKernel<3> > : public Base_FMM_Operator<std::complex<double> > 
    {
        typedef CombinedFieldKernel<3>  KERNEL;
        typedef CombinedFieldHelmholtzIfgfOperator<3> OperatorType;
        typedef Base_FMM_Operator<std::complex<double>> BASE;

    protected:
        std::unique_ptr<OperatorType> op;
        KERNEL kernel;

    public:
        IFGF_Operator(KERNEL _kernel, Array<Vec<3> > _xpts, Array<Vec<3> > _ypts,
                Array<Vec<3>> _xnv, Array<Vec<3>> _ynv,const FMM_Parameters& fmm_params)
        : BASE(std::move(_xpts), std::move( _ypts), std::move(_xnv), std::move(_ynv), IVec<2>(), fmm_params),
            kernel(_kernel)
        {
        std::cout<<"creating ifgf cf op"<<std::endl;
        double waveNumber=_kernel.GetKappa();
        size_t leafSize=250;
        size_t order=8;
        int n_elem=1;
        double tol=-1;
        
        std::cout<<"size="<<xpts.Size()<<std::endl;
        std::cout<<"size="<<ypts.Size()<<std::endl;


        op=make_unique<CombinedFieldHelmholtzIfgfOperator<3> > (waveNumber,leafSize,order,n_elem,tol);
        
        auto srcs=Eigen::Map<typename OperatorType::PointArray>( xpts[0].Data(),3, xpts.Size());
        auto targets=Eigen::Map<typename OperatorType::PointArray>(ypts[0].Data(),3, ypts.Size());
        
        auto src_normals=Eigen::Map<typename OperatorType::PointArray>(xnv[0].Data(),3, xnv.Size());
        
        op->init(srcs,targets,src_normals);
        }


        void  Mult(const BaseVector & x, BaseVector & y) const 
        {
        std::cout<<"ifgf mult"<<std::endl;
        static Timer tall("ngbem fmm apply CombinedField (IFGF)"); RegionTimer reg(tall);
        auto fx = x.FV<Complex>();
        auto fy = y.FV<Complex>();

        //fy = 0;


        auto weights=Eigen::Map< Eigen::Vector<std::complex<double>, Eigen::Dynamic> >(fx.Data(),fx.Size());
        auto results=op->mult(weights);


        auto y_map=Eigen::Map< Eigen::Vector<std::complex<double>, Eigen::Dynamic> >(fy.Data(),fy.Size());
        y_map=results;
        }

    };

    template <>
    class IFGF_Operator<LaplaceSLKernel<3>> : public Base_FMM_Operator<double>
    {
        typedef LaplaceSLKernel<3> KERNEL;
        typedef LaplaceIfgfOperator<3> OperatorType;
        typedef Base_FMM_Operator<double> BASE;

    protected:
        std::unique_ptr<OperatorType> op;
        KERNEL kernel;
        // Array<Vec<3>> xpts, ypts, xnv, ynv;
    public:
        IFGF_Operator(KERNEL _kernel, Array<Vec<3>> _xpts, Array<Vec<3>> _ypts,
                      Array<Vec<3>> _xnv, Array<Vec<3>> _ynv, const FMM_Parameters &fmm_params)
            : BASE(std::move(_xpts), std::move(_ypts), std::move(_xnv), std::move(_ynv), IVec<2>(), fmm_params),
              kernel(_kernel)
        {
            std::cout << "creating ifgf op" << std::endl;

            size_t leafSize = 250;
            size_t order = 10;
            int n_elem = 1;
            double tol = -1;

            std::cout << "size=" << xpts.Size() << std::endl;
            std::cout << "size=" << ypts.Size() << std::endl;

            op = make_unique<LaplaceIfgfOperator<3>>(leafSize, order, n_elem, tol);

            auto srcs = Eigen::Map<typename OperatorType::PointArray>(xpts[0].Data(), 3, xpts.Size());
            auto targets = Eigen::Map<typename OperatorType::PointArray>(ypts[0].Data(), 3, ypts.Size());

            op->init(srcs, targets);
        }

        void Mult(const BaseVector &x, BaseVector &y) const
        {
            std::cout << "ifgf mult" << std::endl;
            static Timer tall("ngbem fmm apply LaplaceSL (IFGF)");
            RegionTimer reg(tall);
            auto fx = x.FV<double>();
            auto fy = y.FV<double>();

            // fy = 0;

            auto weights = Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(fx.Data(), fx.Size());
            auto results = op->mult(weights);

            auto y_map = Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>>(fy.Data(), fy.Size());
            y_map = results;
            // y *= 1.0 / (4*M_PI);
        }
    };
    */

#endif

}

#endif

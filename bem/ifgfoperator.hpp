#ifndef FILE_IFGFOPERATOR
#define FILE_IFGFOPERATOR

#include "fmmoperator.hpp"
//#include <helmholtz_ifgf.hpp>
#include <modified_helmholtz_ifgf.hpp>
#include <double_layer_helmholtz_ifgf.hpp>
#include <combined_field_helmholtz_ifgf.hpp>
#include <laplace_ifgf.hpp>
#include <Eigen/Dense>

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
        typedef ModifiedHelmholtzIfgfOperator<3> OperatorType;
        typedef Base_FMM_Operator<std::complex<double>> BASE;

    protected:
        std::unique_ptr<OperatorType> op;
        KERNEL kernel;

    public:
        IFGF_Operator(KERNEL _kernel, Array<Vec<3>> _xpts, Array<Vec<3>> _ypts,
                      Array<Vec<3>> _xnv, Array<Vec<3>> _ynv, const FMM_Parameters &fmm_params)
            : BASE(std::move(_xpts), std::move(_ypts), std::move(_xnv), std::move(_ynv), KERNEL::Shape(), fmm_params),
              kernel(_kernel)
        {
            std::cout << "creating ifgf helmholtz SL op" << std::endl;
           
            
                std::complex<double> waveNumber = -1i * _kernel.GetKappa();
                std::cout << waveNumber << std::endl;
                size_t leafSize = 250;
                size_t order = 10;
                int n_elem = 1;
                double tol = -1;

                std::cout << "size=" << xpts.Size() << std::endl;
                std::cout << "size=" << ypts.Size() << std::endl;

                op = make_unique<ModifiedHelmholtzIfgfOperator<3>>(waveNumber, leafSize, order, n_elem, tol);

                auto srcs = Eigen::Map<typename OperatorType::PointArray>(xpts[0].Data(), 3, xpts.Size());
                auto targets = Eigen::Map<typename OperatorType::PointArray>(ypts[0].Data(), 3, ypts.Size());
                
                std::function<bool(double dist)> cutOff = [waveNumber](const double dist)
                { return (exp(-waveNumber.real() * dist)) < 1e-7; };
                op->init(srcs, targets, cutOff);
            
        }

        void Mult(const BaseVector &x, BaseVector &y) const
        {
            std::cout << "ifgf mult SL" << std::endl;
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

#endif

}

#endif

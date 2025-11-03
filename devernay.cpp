
//#define GTest

#ifndef FALSE
#define FALSE 0
#endif /* !FALSE */

#ifndef TRUE
#define TRUE 1
#endif /* !TRUE */
#include "devernay.h"

/*----------------------------------------------------------------------------*/
/* fatal error, print a message to standard error and exit
 */
static void error(std::string str)
{
    std::cerr << "Error" << str << std::endl;
    std::cout << "erro " << str << std::endl;
    exit(EXIT_FAILURE);
}





/*----------------------------------------------------------------------------*/
/* compute the image gradient, giving its x and y components as well as the
   modulus. Gx, Gy, and modG must be already allocated.
 */
static void compute_gradient(cv::Mat& Gx, cv::Mat& Gy, cv::Mat& modG,
    cv::Mat& image, int X, int Y) {

    /* check input */
    if (Gx.data == NULL || Gy.data == NULL || modG.data == NULL || image.data == NULL)
        error("compute_gradient: invalid input");


    /* approximate image gradient using centered differences  后一列减前一列  后一行减前一行 计算图像的梯度*/
    /*
           中心差分
           Gx​(x,y)=I(x+1,y)−I(x−1,y)
           Gy​(x,y)=I(x,y+1)−I(x,y−1)
    */
    for (int x = 1; x < X - 1; x++) {
        for (int y = 1; y < Y - 1; y++) {

            Gx.at<double>(y, x) = image.at<double>(y, x + 1) - image.at<double>(y, x - 1);
            Gy.at<double>(y, x) = image.at<double>(y + 1, x) - image.at<double>(y - 1, x);
            modG.at<double>(y, x) = sqrt(std::pow(Gx.at<double>(y, x), 2) + std::pow(Gy.at<double>(y, x), 2));
        }

    }


}
/*----------------------------------------------------------------------------*/
/* compute a Gaussian kernel of length n, standard deviation sigma,
   and centered at value mean.

   for example, if mean=0.5, the Gaussian will be centered in the middle point
   between values kernel[0] and kernel[1].

   kernel must be allocated to a size n.

   生成卷积核
 */
static void gaussian_kernel(cv::Mat& kernel, int n, double sigma, double mean) {

    double sum = 0.0;
    double val;

    /* check input */
    if (kernel.data == NULL)
        error("gaussian_kernel: kernel not allocated");
    if (sigma <= 0.0)
        error("gaussian_kernel: sigma must be positive");
    /* compute Gaussian kernel */
    for (int i = 0; i < n; i++) {

        val = (static_cast<double>(i) - mean) / sigma;
        kernel.at<double>(0, i) = exp(-0.5 * val * val);
        sum += kernel.at<double>(0, i);
    }

    /* normalization */
    if (sum > 0.0) {
        for (int i = 0; i < n; i++) {
            kernel.at<double>(0, i) /= sum;
        }
    }


}


/*----------------------------------------------------------------------------*/
/* filter an image with a Gaussian kernel of parameter sigma. return a pointer
   to a newly allocated filtered image, of the same size as the input image.
 */
static void gaussian_filter(cv::Mat& image, int X, int Y, double sigma, cv::Mat& grad_image) {



    /* check input */
    if (sigma <= 0.0)
        error("gaussian_filter: sigma must be positive");
    if (image.data == NULL || X < 1 || Y < 1)
        error("gaussian_filter: invalid image");

    /* compute gaussian kernel */
    /*
        The size of the kernel is selected to guarantee that the first discarded
        term is at least 10^prec times smaller than the central value. For that,
        the half size of the kernel must be larger than x, with
          e^(-x^2/2sigma^2) = 1/10^prec
        Then,
          x = sigma * sqrt( 2 * prec * ln(10) )  ----- offset
          截断的位置（即“第一个被舍弃的项”）的值, 要比核中心值（最大值）小10^prec 倍这样才能保证数值误差很小。
    */

    double prec = 3.0;
    int offset = (int)ceil(sigma * sqrt(2.0 * prec * log(10.0)));
    int n = 1 + 2 * offset;   /* kernel size */

    cv::Mat kernel(1, n, CV_64F);        // 用于存储卷积核  高斯卷积是一种可分离卷积核
    gaussian_kernel(kernel, n, sigma, static_cast<double>(offset));


    // 辅助变量 用于当图像 size*2
    int nx2 = 2 * X;
    int ny2 = 2 * Y;
    double val;
    cv::Mat tmp(Y, X, CV_64F);
    cv::Mat out(Y, X, CV_64F);

    // 执行卷积操作  Gx
    for (int x = 0; x < X; x++) {
        for (int y = 0; y < Y; y++) {
            val = 0.0;
            for (int i = 0; i < n; i++) {
                int j = x - offset + i;
                /* symmetry boundary condition  边界条件*/
                while (j < 0) j += nx2;
                while (j >= nx2) j -= nx2;
                if (j >= X) j = nx2 - 1 - j;

                val += image.at<double>(y, j) * kernel.at<double>(0, i);
            }
            tmp.at<double>(y, x) = val;
        }

    }

    // 执行卷积操作  Gy
    for (int x = 0; x < X; x++) {
        for (int y = 0; y < Y; y++) {
            val = 0.0;
            for (int i = 0; i < n; i++) {
                int j = y - offset + i;

                /* symmetry boundary condition */
                while (j < 0) j += ny2;
                while (j >= ny2) j -= ny2;
                if (j >= Y) j = ny2 - 1 - j;

                val += tmp.at<double>(j, x) * kernel.at<double>(0, i);
            }
            out.at<double>(y, x) = val;
        }
    }


#ifdef GTest
    cv::Mat mag8U;
    cv::normalize(out, mag8U, 0, 255, cv::NORM_MINMAX);
    mag8U.convertTo(mag8U, CV_8U);
    cv::imwrite("gradient_magnitude.png", mag8U);
#endif // GTest

    out.copyTo(grad_image);
    // 关于cv::Mat 之间的拷贝
    /*
        cv::Mat A , B;
        A = B; 浅拷贝  两者共用同一块地址
        B.copyTo(A);  深拷贝   比clone更灵活
    */
}



/*----------------------------------------------------------------------------*/
/* compute a > b considering the rounding errors due to the representation
   of double numbers
 */
static int greater(double a, double b)
{
    if (a <= b) return FALSE;  /* trivial case, return as soon as possible */

    if ((a - b) < 1000 * DBL_EPSILON) return FALSE;

    return TRUE; /* greater */
}

static void compute_edge_points(cv::Mat& Ex, cv::Mat& Ey, cv::Mat& modG,
    cv::Mat& Gx, cv::Mat& Gy, int X, int Y) {

    /* check input */
    if (Ex.data == NULL || Ey.data == NULL || modG.data == NULL || Gx.data == NULL || Gy.data == NULL)
        error("compute_edge_points: invalid input");

    Ex.setTo(-1.0), Ey.setTo(-1.0);   // 记录是否为边缘点

    /* explore pixels inside a 2 pixel margin (so modG[x,y +/- 1,1] is defined) */
    /*
    * 当梯度模值满足局部水平方向极大值且梯度更接近水平时 |Gx| >= |Gy|  判定为 水平 边缘
    * 当梯度值模满足局部垂直方向极大值且梯度方向更接近垂直时|Gx| <= |Gy|  判断为 垂直 边缘
    * 可能出现相邻两像素相等且同为极大值的情况， 例如边缘恰好位于两像素之间
    * 根据预设规则， 水平极大值标记为左侧点  垂直极大值标记下方像素
    * 对应的判断条件为： 左临点 < 当前模值 >= 右临点   下临点 < 当前模值 >= 上临点
    * 比较是通过greater() 函数实现， 旨在将因舍入产生差异的数值视作相等
    */

    for (int x = 2; x < X - 2; x++) {
        for (int y = 2; y < Y - 2; y++) {

            int Dx = 0;            /* interpolation will be along Dx,Dy */
            int Dy = 0;            /*   which will be selected below    */
            double mod = modG.at<double>(y, x);       /* modG at pixel              */
            double L = modG.at<double>(y, x - 1);     /* modG at pixel on the left  */
            double R = modG.at<double>(y, x + 1);     /* modG at pixel on the right */
            double U = modG.at<double>(x + 1, y);     /* modG at pixel up           */
            double D = modG.at<double>(x - 1, y);     /* modG at pixel below        */

            double gx = fabs(Gx.at<double>(y, x));
            double gy = fabs(Gy.at<double>(y, x));

            if (greater(mod, L) && !greater(R, mod) && gx >= gy) Dx = 1;  // 水平     判断是水平方向极大值  还是竖直方向上的极大值    
            else if (greater(mod, D) && !greater(U, mod) && gx <= gy) Dy = 1;  // 竖直

            // step2： Devernay sub-pixel correction [2]
            /*
            * 边缘点位置的确定是通过沿一维方向对梯度值进行二次插值， 并选取插值函数的及大值点， 该像素必须是局部极大值点
            * 已知三点的梯度模值分别为
            *                                          . b
                                                a .    |
                 x = -1, |Gx| = a                 |    |    . c
                 x =  0, |Gx| = b                 |    |    |
                 x =  1, |Gx| = c               ------------------> x
                                                 -1    0    1
              穿过点(-1,a)、(0,b)和(1,c)的抛物线，其极大值点的x坐标为：
              偏移量 = (a - c) / [2(a - 2b + c)]
              由于 b >= a 且 b >= c，可得 -0.5 <= 偏移量 <= 0.5
            */

            if (Dx > 0 || Dy > 0) {
                /* offset value is in [-0.5, 0.5]   亚像素坐标小于一个整像素*/
                double a = modG.at<double>(y - Dy, x - Dx);
                double b = modG.at<double>(y, x);
                double c = modG.at<double>(y + Dy, x + Dx);

                double offset = 0.5 * (a - c) / (a - b - b + c);

                // 存储亚像素值
                Ex.at<double>(y, x) = x + offset * Dx;
                Ey.at<double>(y, x) = y + offset * Dy;

            }
        }
    }
#ifdef GTest
    cv::FileStorage fesx("Ex.yml", cv::FileStorage::WRITE);
    fesx << "Ex" << Ex;
    fesx.release();

    cv::FileStorage fesy("Ey.yml", cv::FileStorage::WRITE);
    fesy << "Ey" << Ey;
    fesy.release();
#endif // GTest




}


/*----------------------------------------------------------------------------*/
/* return a score for chaining pixels 'from' to 'to', favoring closet point:
   = 0.0 invalid chaining
   > 0.0 valid forward chaining; the larger the value, the better the chaining
   < 0.0 valid backward chaining; the smaller the value, the better the chaining

   input:
     from, to       the two pixel IDs to evaluate their potential chaining
     Ex[i], Ey[i]   the sub-pixel position of point i, if i is an edge point;
                    they take values -1,-1 if i is not an edge point
     Gx[i], Gy[i]   the image gradient at pixel i
     X, Y           the size of the image
 */
static double chain(cv::Point2i from, cv::Point2i to, cv::Mat& Ex, cv::Mat& Ey,
    cv::Mat& Gx, cv::Mat& Gy, int X, int Y) {

    double dx, dy;

    /* check input */
    if (Ex.data == NULL || Ey.data == NULL || Gx.data == NULL || Gy.data == NULL)
        error("chain: invalid input");
    if (from.x < 0 || to.x < 0 || from.x * from.y >= X * Y || to.x * to.y >= X * Y)
        error("chain: one of the points is out the image");

    /* check that the points are different and valid edge points,
      otherwise return invalid chaining */
    if (from == to) return 0.0; /* same pixel, not a valid chaining */
    if (Ex.at<double>(from.y, from.x) < 0.0 || Ey.at<double>(from.y, from.x) < 0.0 || Ex.at<double>(to.y, to.x) || Ey.at<double>(to.y, to.x))
        return 0.0;  /* one of them is not an edge point, not a valid chaining */

    /* 在一个好的链接关系中， 梯度方向应该与待连接两点连接方向大致正交

              Gx,Gy
             |                        ------> dx,dy
             |               thus
        from x-------x to             ---> Gy,-Gx  (orthogonal to the gradient)

      当 Gy * dx - Gx * dy > 0 时，对应前向连接；
      当 Gy * dx - Gx * dy < 0 时，对应后向连接。
      (人为设计)

      首先验证待连接两点的梯度方向是否一致， 否则返回无效连接
    */

    dx = Ex.at<double>(to.y, to.x) - Ex.at<double>(from.y, from.x);
    dy = Ey.at<double>(to.y, to.x) - Ey.at<double>(from.y, from.x);

    if ((Gy.at<double>(from.y, from.x) * dx - Gx.at<double>(from.y, from.x) * dy) *
        (Gy.at<double>(to.y, to.x) * dx - Gx.at<double>(to.y, to.x) * dy) <= 0.0) {
        return 0.0;
    }
}



/*----------------------------------------------------------------------------*/
/* chain edge points

   input: Ex and Ey are the sub-pixel coordinates when an edge point is present
          or -1,-1 otherwise. Gx, Gy and modG are the x and y components and the
          modulus of the image gradient, respectively. X,Y is the image size.

   output: next and prev will contain the number of next and previous edge
           points in the chain. when not chained in one of the directions, the
           corresponding value is set to -1. next and prev must be allocated
           before calling.
 */
static void chain_edge_points(int* next, int* prev, cv::Mat& Ex, cv::Mat& Ey,
    cv::Mat& Gx, cv::Mat& Gy, int X, int Y) {

    /* check input */
    if (next == NULL || prev == NULL || Ex.data == NULL || Ey.data == NULL || Gx.data == NULL || Gy.data == NULL)
        error("chain_edge_points: invalid input");

    /* initialize next and prev as non linked */
    for (int i = 0; i < X * Y; i++) next[i] = prev[i] = -1;

    /* try each point to make local chains */
    for (int x = 2; x < X; x++) {
        for (int y = 2; y < Y; y++) {
            if (Ex.at<double>(y, x) >= 0 && Ey.at<double>(y, x) >= 0) {  /* must be an edge point  该方法只插值水平与竖直两个方向*/

                // int from = x + y * X;
                cv::Point2i from(x, y);    // cv::Point(x, y) 的语义是 “x = 列（宽度方向）”，“y = 行（高度方向
                double fwd_s = 0.0;  /* score of best forward chaining */
                double bck_s = 0.0;  /* score of best backward chaining */
                int fwd = -1;        /* edge point of best forward chaining */
                int bck = -1;        /* edge point of best backward chaining */

                /*
                * 便利所有距离两像素以及内的邻接点， 在大多数情况下，寻找两像素间距的连接候选点
                * 足以构建能精确描述边缘的优质边缘点
                */

                for (int i = -2; i <= 2; i++) {
                    for (int j = -2; j <= 2; j++) {
                        // int to = x + i + (y + j) * X;   // 待连接边缘点的候选
                        cv::Point2i to(x + i, y + j);

                    }
                }
            }
        }
    }


}



/*----------------------------------------------------------------------------*/
/* chained, sub-pixel edge detector. based on a modified Canny non-maximal
   suppression and a modified Devernay sub-pixel correction.

   input:

     image        : the input image
     X,Y          : the size of the input image
     sigma        : standard deviation sigma for the Gaussian filtering
                    (if sigma=0 no filtering is performed)
     th_h         : high gradient threshold in Canny's hysteresis
     th_l         : low gradient threshold in Canny's hysteresis

   output:

     x,y          : lists of sub-pixel coordinates of edge points
     curve_limits : the limits of each curve in lists x and y
     N            : number of edge points
     M            : number of curves

   the input is a XxY graylevel image given as a pointer to an array of doubles
   such that image[x+y*X] is the value at coordinates x,y
   (for 0 <= x < X and 0 <= y < Y).

   the output are the chained edge points given as 3 allocated lists: x, y and
   curve_limits. also the numbers N (size of lists x and y) and M (number of
   curves).

   x[i] and y[i] (0<=i<N) store the sub-pixel coordinates of the edge points.
   curve_limits[j] (0<=j<=M) stores the limits of each chain in lists x and y.

   example:

     curve number k (0<=k<M) consists of the edge points x[i],y[i]
     for i determined by curve_limits[k] <= i < curve_limits[k+1].

     curve k is closed if x[curve_limits[k]] == x[curve_limits[k+1] - 1] and
                          y[curve_limits[k]] == y[curve_limits[k+1] - 1].
 */

void devernay(double** x, double** y, int* N, int** curve_limits, int* M,    // 输入输出
    cv::Mat& image, int X, int Y,
    double sigma, double th_h, double th_l) {

    // step1： 创建存储结果的中间变量

    cv::Mat Gx(Y, X, CV_64F);        /* grad_x */
    cv::Mat Gy(Y, X, CV_64F);        /* grad_y */
    cv::Mat modG(Y, X, CV_64F);      /* |grad| */

    // step2: 生成卷积核  对图像进行卷积
    cv::Mat gauss;
    if (sigma == 0.0) compute_gradient(Gx, Gy, modG, image, X, Y);
    else
    {
        gaussian_filter(image, X, Y, sigma, gauss);
        //cv::imshow("123", gauss);
        //cv::waitKey();
        compute_gradient(Gx, Gy, modG, gauss, X, Y);
    }

#ifdef GTest
    cv::FileStorage fsx("Gx.yml", cv::FileStorage::WRITE);
    fsx << "Gx" << Gx;
    fsx.release();

    cv::FileStorage fsy("Gy.yml", cv::FileStorage::WRITE);
    fsy << "Gy" << Gy;
    fsy.release();
#endif // GTest


    cv::Mat next(Y, X, CV_32SC1);   /* next point in chain */
    cv::Mat prev(Y, X, CV_32SC1);    /* prev point in chain */

    cv::Mat Ex(Y, X, CV_64F);        /* edge_x */
    cv::Mat Ey(Y, X, CV_64F);        /* edge_y */
    compute_edge_points(Ex, Ey, modG, Gx, Gy, X, Y);
}
// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
using namespace arma;
using namespace std;

// #define C0 1/(13/exp(3)-40/exp(2)+13/datum::e)
#define C0 1

arma::vec Kernel(const arma::vec& x, const double& h){
  // normal Kernel
  return normpdf(x/h);
}

double EstimateUV(const arma::vec& Z, const arma::vec& Y, const int& i, const double& h){
  return sum(dot(Kernel(Z-Z(i),h), (Y<=Y(i)))) / sum(Kernel(Z-Z(i),h));
}


double EstimateW(const arma::vec& Z, const int& i) {
  return sum(Z <= Z(i)) / (Z.size()+0.0); // +0.0 to convert int to double; or else int/int gives int; double/int or int/double gives double
}

//' Estimate rho
//'
//' @param U A vector of U
//' @param V A vector of V
//' @param W A vector of W
//' @export
// [[Rcpp::export]]
double EstimateRho(const arma::vec& U, const arma::vec& V, const arma::vec& W){
  double temp = 0;
  int n = U.size();
  for(int i=0;i<n;i++){
    for(int j=i+1;j<n;j++){
      temp += (exp(-abs(U(i)-U(j))) + exp(-U(i))+exp(U(i)-1)+exp(-U(j))+exp(U(j)-1)+2*exp(-1)-4) * (exp(-abs(V(i)-V(j))) + exp(-V(i))+exp(V(i)-1)+exp(-V(j))+exp(V(j)-1)+2 * exp(-1) -4) * exp(-abs(W(i)-W(j)));
    }
  }
  temp *= 2;
  /*for(int i=0;i<n;i++){
   temp += (1 + 2*exp(-U(i))+2*exp(U(i)-1)+2 * exp(-1) -4) * (1 + 2*exp(-V(i))+2*exp(V(i)-1)+2 * exp(-1)-4);
  }*/
  //return temp*C0/pow(n,2);
  return temp * C0 / (n*(n-1));
}

//' Estimate the CI test statistics
//'
//' @param X A matrix of covariates
//' @param Y A vector of response
//' @param Z A vector of conditional variable
//' @param h kernel bandwidth
//' @export
// [[Rcpp::export]]
arma::vec CITestStat(const arma::mat& X, const arma::vec& Y, const arma::vec& Z, const double& h){
  int n = X.n_rows;
  int p = X.n_cols;

  arma::vec rho = vec(p);
  arma::vec V = vec(n);
  arma::vec W = vec(n);
  arma::vec U = vec(n);

  for(int i=0; i<n; i++){
    V(i) = EstimateUV(Z,Y,i,h);
    W(i) = EstimateW(Z,i);
  }

  for(int k=0;k<p;k++){
    for(int i=0; i<n; i++){
      U(i) = EstimateUV(Z,X.col(k),i,h);
    }
    rho(k) = EstimateRho(U,V,W);
  }
  return rho;
}


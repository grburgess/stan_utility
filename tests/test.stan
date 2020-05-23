data {
}
parameters {
    real<lower=0,upper=1> x;
    real y[10];
}
model {
    y ~ normal(1, 2);
}

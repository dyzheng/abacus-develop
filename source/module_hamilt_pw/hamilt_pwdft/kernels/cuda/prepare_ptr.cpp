// cal_vkb
{
    thrust::complex<FPTYPE>** vkb_ptrs[nh];
    const FPTYPE** ylm_ptrs[nh];
    const FPTYPE** vq_ptrs[nh];
    int ih = 0;
    for (int nb = 0; nb < nbeta; nb++)
    {
        int l = nhtol[it * nhtol_nc + ih];
        for (int m = 0; m < 2 * l + 1; m++)
        {
            int lm = l * l + m;
            vkb_ptrs[ih] = &vkb_out[ih * npw];
            ylm_ptrs[ih] = &ylm_in[lm * npw];
            vq_ptrs[ih] = &vq_in[nb * npw];
            ih++;
        }
    }
}

// cal_vkb_deri
{
    thrust::complex<FPTYPE>** vkb_ptrs[nh];
    const FPTYPE** ylm_ptrs[nh];
    const FPTYPE** vq_ptrs[nh];

    const FPTYPE** ylm_deri_ptr1s[nh];
    const FPTYPE** ylm_deri_ptr2s[nh];
    const FPTYPE** vq_deri_ptrs[nh];

    int ih = 0;
    int x1 = (GlobalC::ppcell.lmaxkb + 1) * (GlobalC::ppcell.lmaxkb + 1);
    for (int nb = 0; nb < nbeta; nb++)
    {
        int l = nhtol[it * nhtol_nc + ih];
        for (int m = 0; m < 2 * l + 1; m++)
        {
            int lm = l * l + m;
            vkb_ptrs[ih] = &vkb_out[ih * npw];
            ylm_ptrs[ih] = &ylm_in[lm * npw];
            vq_ptrs[ih] = &vq_in[nb * npw];

            ylm_deri_ptr1s[ih] = &ylm_deri_in[(ipol * x1 + lm) * npw];
            ylm_deri_ptr2s[ih] = &ylm_deri_in[(jpol * x1 + lm) * npw];
            vq_deri_ptrs[ih] = &vq_deri_in[nb * npw];

            ih++;
        }
    }
}

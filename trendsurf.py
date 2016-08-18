#!/home/mrstats/maamen/Software/python/bin/python2.7

#------------------------------------------------------------------------------
# Usage: 
# python trendsurf.py -m [maskfile] -b [basis] -c [covariates] <infile> 
#
# Written by A. Marquand
#------------------------------------------------------------------------------

from __future__ import print_function

import os
import numpy as np

# ------------------------------ Load data -----------------------------------
def load_data(datafile, maskfile):  
    import nibabel as nib
        
    # load 4d nifti data
    if datafile.endswith("nii.gz") or datafile.endswith("nii"):
        print('Loading nifti input ...')
        img = nib.load(datafile)
        dat = img.get_data()
        dim = dat.shape[0:3]
    else:
        print('No routine to handle non-nifti data')
        raise ValueError
      
    # load a nifti mask
    if maskfile is None:
        print('No mask specified. Generating one automatically ...')
        mask = dat[:,:,:,0] != 0        
    else:         
        print('Loading ROI mask ...')
        if maskfile.endswith("nii.gz") or maskfile.endswith(".nii"):
            imm = nib.load(maskfile)
            mask = imm.get_data() != 0
        else:
            print("Invalid mask filename specified")
            raise ValueError
            
    maskid = np.where(mask.flatten())[0]

    # mask the image      
    dat = np.reshape(dat, (np.prod(dim), dat.shape[3]))
    dat = dat[maskid,:]     
    
    # generate voxel coordinates
    i,j,k = np.meshgrid(np.linspace(0,dim[0]-1,dim[0]),
                        np.linspace(0,dim[1]-1,dim[1]),
                        np.linspace(0,dim[2]-1,dim[2]),indexing='ij') 
                        
    # voxel-to-world mapping
    world = np.vstack((i.ravel(), j.ravel(), k.ravel(), 
                       np.ones(np.prod(i.shape),float) )).T
    world = np.dot(world,img.affine.T)[maskid,0:3]
  
    return dat, world, mask
 
# --------------------------Polynomial basis set-------------------------------
def create_basis(X,basis):

    # check whether we are using a polynomial basis set
    if type(basis) is int or (type(basis) is str and len(basis) == 1):
        dimpoly = int(basis)
        dimx = X.shape[1]
        print('Generating polynomial basis set of degree',dimpoly,'...')
        Phi = np.zeros((X.shape[0], X.shape[1]*dimpoly))
        colid = np.arange(0,dimx)
        for d in range(1,dimpoly+1):
            Phi[:,colid] = X ** d
            colid += dimx
    else: # custom basis set
        print("Custom basis set is not implemented yet!")
        raise ValueError
    
    return Phi
   
# --------------------------- Write nifti image -------------------------------   
def write_nii(data,filename,examplenii,mask):
    import nibabel as nib
    
    ex_img = nib.load(examplenii)
    dim = ex_img.shape[0:3]
    nvol = int(data.shape[1])   
    
    array_data = np.zeros((np.prod(dim),nvol))
    array_data[mask.flatten(),:] = data
    array_data = np.reshape(array_data,dim+(nvol,))
    array_img = nib.Nifti1Image(array_data, ex_img.get_affine(),ex_img.get_header())
    nib.save(array_img,filename)  
    
# -----------------------------Main routine ----------------------------------
def main(*args):
    import argparse
    from bayesreg import blr
    
    np.seterr(invalid='ignore')
    
    # parse arguments
    parser = argparse.ArgumentParser(description="Trend surface model")
    parser.add_argument("filename")
    parser.add_argument("-b", help="basis set",dest="basis",default=3)
    parser.add_argument("-m", help="mask file",dest="maskfile",default=None)
    parser.add_argument("-c", help="covariates file",dest="covfile",default=None)
    args = parser.parse_args()
    wdir = os.path.realpath(os.path.curdir)
    filename = os.path.join(wdir,args.filename)
    if args.maskfile is None:
        maskfile = None
    else:
        maskfile = os.path.join(wdir,args.maskfile)
    basis = args.basis
    if args.covfile is not None:
        print("Ignoring covariate information (not implemented yet).")
    
    # load data
    print("Processing data in",filename)   
    Y,X,mask = load_data(filename,maskfile)   
    Y = np.round(10000*Y)/10000 # truncate precision to avoid numerical problems
    N = Y.shape[1]
    
    # standardize responses and covariates
    mY = np.mean(Y,axis=0)
    sY = np.std(Y,axis=0)
    Yz = (Y - mY) / sY
    mX = np.mean(X,axis=0)
    sX = np.std(X,axis=0)
    Xz = (X - mX) / sX
    
    # create basis set and set starting hyperparamters
    Phi = create_basis(Xz,basis)
    hyp0 = np.zeros(2)  
    
    # estimate the data from all subjects
    yhat = np.zeros_like(Yz)
    ys2 = np.zeros_like(Yz)
    nlZ = np.zeros(N)
    hyp = np.zeros((N,len(hyp0)))
    rmse = np.zeros(N)
    ev = np.zeros(N)
    m = np.zeros((N,Phi.shape[1]))
    for i in range(0,N):
        print("Estimating model ",i+1,"of",N)
        breg = blr()
        hyp[i,:] = breg.estimate(hyp0, Phi, Yz[:,i])
        m[i,:] = breg.m
        nlZ[i] = breg.nlZ
        
        # compute predictions and errors
        yhat[:,i],ys2[:,i] = breg.predict(hyp[i,:], Phi, Yz[:,i], Phi)
        yhat[:,i] = yhat[:,i]*sY[i] + mY[i]
        rmse[i] = np.sqrt(np.mean((Y[:,i] - yhat[:,i]) ** 2))
        ev[i] = 100*(1 - (np.var(yhat[:,i] - Y[:,i]) / np.var(Y[:,i])))
    
        print("Variance explained =",ev[i], "% RMSE =",rmse[i])       

    print("Mean (std) variance explaind = ",ev.mean(),"(",ev.std(),")")
    print("Mean (std) RMSE = ",rmse.mean(),"(",rmse.std(),")")
    
    # Write output
    print("Writing output ...")
    np.savetxt("trendcoeff.txt",m,delimiter='\t',fmt='%5.8f')
    np.savetxt("negloglik.txt",nlZ,delimiter='\t',fmt='%5.8f') 
    np.savetxt("hyp.txt",hyp,delimiter='\t',fmt='%5.8f')
    np.savetxt("explainedvar.txt",ev,delimiter='\t',fmt='%5.8f')
    np.savetxt("rmse.txt",rmse,delimiter='\t',fmt='%5.8f')  
    write_nii(yhat,'yhat.nii.gz',filename,mask)
    write_nii(ys2,'ys2.nii.gz',filename,mask)
    
# For running from the command line:
if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
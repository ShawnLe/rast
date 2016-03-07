/* Copyright (c) 1990-1995 by Thomas M. Breuel */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>


#include "misc.h"
#include "narray.h"
#include "vec2.h"

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "EdgeDetector.hpp"

using namespace colib;

#include "util.h"
#include "rast.h"

namespace lumo_cinstancep2d {

/// a point with angle value
struct avec {
  vec2 p;
  float a;
};

typedef avec Msource;
typedef avec Mpoint;
typedef avec Ipoint;

inline int urand48() { return abs(int(lrand48())); }

#if 0
    template <class T>
    static void shuffle(narray<T> &narray) {
        int n = narray.length();
        for(int i=0;i<n-1;i++) {
            int j = urand48() % (n-i) + i;
            if(i!=j) swap(narray[i],narray[j]);
        }
    }
#endif

struct CInstanceP2D : InstanceP2D {
  int image_size;
  int model_size;

  int nclutter;
  /// number of model points
  int nmodel_total;
  int nmodel_unoccluded;
  float error;
  /// angle error
  float aerror;
  float minscale;
  float maxscale;

  vec2 translation;
  float angle;
  float scale;
  float get_param(int i) {
    switch (i) {
      case 0:
        return translation[0];
      case 1:
        return translation[1];
      case 2:
        return angle;
      case 3:
        return scale;
      default:
        throw "parameter index out of range";
    }
  }
  /// model point array 
  narray<Msource> msources;
  /// image point array
  narray<Ipoint> ipoints;

  CInstanceP2D() { init(); }

  void init() {
    image_size = 512;
    model_size = 100;
    nclutter = igetenv("nclutter", 50);
    nmodel_total = 20;
    nmodel_unoccluded = 10;
    error = 5.0;
    aerror = 0.1;
  }

  
  /**
   * @author Shawn Le 
   * @date   07/Mar/2016
   * @brief  read input data instead of using generated random data
   */ 
  int readInputs()
  {
    int ret = 0;
    
    cv::Mat modelimg, imageimg; 
    modelimg = cv::imread("model.bmp");
    if (modelimg.empty())
    {
      std::cout << "error! model cannot retrieve" << std::endl;
      ret = 1;
      return ret;
    }
    
    imageimg = cv::imread("image.bmp");
    if (imageimg.empty())
    {
      std::cout << "error! image cannot retrieve" << std::endl;
      ret = 1;
      return ret;
    }
    
    EdgeDetector ed; 
    EdgeDetector::EdgeData modelEdDat, imageEdDat;
    cv::Mat modelEdge, imageEdge;
    ed.EdgeDetect(modelimg, modelEdDat, modelEdge);
    ed.EdgeDetect(imageimg, imageEdDat, imageEdge);
 
    // pass inputs to compatible data format
    msources.clear();
    ipoints.clear();
    nmodel_total = modelEdDat.edgePts.size();
    for (int i = 0; i < nmodel_total; i++) {
      Msource &m = msources.push();
      m.p = vec2((float)modelEdDat.edgePts[i].x, (float)modelEdDat.edgePts[i].y);
      m.a = (float)atan2(modelEdDat.DyArr[i], modelEdDat.DxArr[i]);
    }
    for (int i = 0; i < imageEdDat.edgePts.size(); i++) {
      Ipoint &p = ipoints.push();
      p.p = vec2((float)imageEdDat.edgePts[i].x, (float)imageEdDat.edgePts[i].y);
      p.a = (float)atan2(imageEdDat.DyArr[i], imageEdDat.DxArr[i]);
    }
        
    return ret;
  }
  
  
  /// generate a random test data i.e. model and image
  /**
   * @details 
   * \n 1) model points are randomly created i.e.
   * \n --> m.p vec2(urand(-model_size, model_size), urand(-model_size, model_size))
   * \n --> m.a = urand(0.0, 2 * M_PI)
   * \n 2) projected image points + noise are created 
   * \n --> p.p = cmul(rotation, msources[i].p) + translation + randomUniformVectorFromCircle(error);
   * \n --> p.a = msources[i].a + angle + urand(-aerror, aerror);
   * \n 3) cluttered image points are randomly created
   * @param[out] msources with msources[i].p and msources[i].a
   * @param[out] ipoints  with ipoints[i].p and ipoints[i].a
   */
  void generate() {
    float dx = urand(0.0, image_size);
    float dy = urand(0.0, image_size);
    translation = vec2(dx, dy);
    angle = urand(0.0, 2.0 * M_PI);
    scale = urand(minscale, maxscale);
    vec2 rotation = vec2(scale * cos(angle), scale * sin(angle));
    msources.clear();
    ipoints.clear();
    for (int i = 0; i < nmodel_total; i++) {
      Msource &m = msources.push();
      m.p =
          vec2(urand(-model_size, model_size), urand(-model_size, model_size));
      m.a = urand(0.0, 2 * M_PI);
    }
    for (int i = 0; i < nmodel_unoccluded; i++) {
      Ipoint &p = ipoints.push();
      p.p = cmul(rotation, msources[i].p) + translation +
            randomUniformVectorFromCircle(error);
      p.a = msources[i].a + angle + urand(-aerror, aerror);
    }
    shuffle(msources);
    for (int i = 0; i < nclutter; i++) {
      Ipoint &p = ipoints.push();
      p.p = vec2(urand(0, image_size), urand(0, image_size));
      p.a = urand(0.0, 2 * M_PI);
    }
    shuffle(ipoints);
  }

  void set_image_size(int r) { image_size = r; }
  void set_model_size(int r) { model_size = r; }
  void set_nclutter(int v) { nclutter = v; }
  void set_nmodel_total(int v) { nmodel_total = v; }
  void set_nmodel_unoccluded(int v) { nmodel_unoccluded = v; }
  void set_error(float v) { error = v; }
  void set_aerror(float v) { aerror = v; }
  void set_srange(float min, float max) {
    minscale = min;
    maxscale = max;
  }
  int nimage() { return ipoints.length(); }
  /// @param[out] a -> orientation value e.g. angle of the point
  void get_image(float &x, float &y, float &a, int i) {
    x = ipoints[i].p[0];
    y = ipoints[i].p[1];
    a = ipoints[i].a;
  }
  int nmodel() { return msources.length(); }
  void get_model(float &x, float &y, float &a, int i) {
    x = msources[i].p[0];
    y = msources[i].p[1];
    a = msources[i].a;
  }

  ~CInstanceP2D() {}
};
}

InstanceP2D *makeInstanceP2D() { return new lumo_cinstancep2d::InstanceP2D(); }
CInstanceP2D *makeCInstanceP2D() { return new lumo_cinstancep2d::CInstanceP2D(); }

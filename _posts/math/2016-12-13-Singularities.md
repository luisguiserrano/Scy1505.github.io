---
layout: post
title:  "Singularities"
date:   2016-11-02
category: math
---

My main object of study are singularities, their invariants, and the tools used to study them. Singularities play an important role as they appear naturally in almost every area of mathematics, including commutative algebra, algebraic geometry, number theory, representation theory, analysis and topology. My research is focused mostly on the invariants encountered in positive characteristic and their relation/similarity with invariants appearing in characteristic zero.

## Singularities

From the geometric point of view, a singularity corresponds to a point where a given geometric object, for example a manifold or an algebraic variety, has an unexpected tangent space. For instance, if we consider the following hypersurfaces 

<center>
<table border="0">
<td align="center"><img src="{{ '/assets/img/Singularities_via_Frobenius/Sing1.png' | prepend: site.baseurl }}" alt=""> </td>
<td align="center"><img src="{{ '/assets/img/Singularities_via_Frobenius/Sing2.png' | prepend: site.baseurl }}" alt=""> </td>
<td align="center"><img src="{{ '/assets/img/Singularities_via_Frobenius/Sing3.png' | prepend: site.baseurl }}" alt=""> </td>
</table>
</center>

we can see that in each case there is a special point where the tangent space is larger than expected.

Singularities can also be detected algebraically. For example, if $$f(\boldsymbol{x})$$ is a polynomial in $$n$$ variables with real coefficients defining a hypersurface $$X$$ in $$\mathbb{R}^n$$, then the singularities of $$X$$ are given by those points $$\boldsymbol{a}\in X$$ where the partial derivatives $$\frac{\partial f}{\partial x_i}(\boldsymbol{a})$$ are zero for all $$i$$. For instance,  the hypersurface in the middle of the above figure corresponds to  $$f=z^3-(y^2+3x^2)$$, which has  partial derivatives $$f_x=-6x$$, $$f_y=-2y$$, and $$f_z=3z^2$$. We conclude that the only singular point is the origin, which corresponds with our geometric intuition.  


Classifying singularities has been an object of intense study in both zero and positive characteristic. In characteristic zero many invariants can be described in terms of resolutions of singularities and they are related to the minimal model program. Furthermore, many results in this setting can be approached analytically. However, in the recent decades, it has become apparent that in order to study singularities in characteristic zero, one can also reduce to positive characteristic and use Frobenius techniques to investigate singularities.

## Singularities via Frobenius

 Let $$R$$ be a domain of positive characteristic $$p$$. The Frobenius map $$F:R\to R$$ takes an element $$r$$ to $$r^p$$, therefore its image is the subring $$R^p$$ consisting of all the $$p$$-th powers of elements in $$R$$. This induces on $$R$$ a structure of $$R^p$$-module. It is a consequence of a theorem of Kunz that, under mild conditions, $$R$$ is not singular if and only if $$R$$ is a locally free $$R^p$$-module. This remarkable result tells us that we can detect singularities via the action of Frobenius. Therefore we can define different families of singularities by specifying "how close" $$R$$ is to a locally free $$R^p$$-module. There are many kinds of singularities obtained by imposing restrictions on the action of Frobenius and one key feature is that they parallel the classes of singularities that have been studied in characteristic zero. The following diagram shows the relation among them and how they compare with the singularities in characteristic zero:

 <center>
<img src="{{ '/assets/img/Singularities_via_Frobenius/Sing.png' | prepend: site.baseurl }}" alt="">
 </center>
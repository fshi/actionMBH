// {{{ copyright notice
// }}}
// {{{ file documentation
/**
 * @file
 * @brief Define BiStreamable (Binary Streamable) template based interface
 *
 * @todo Implement BiIStream and BiOStream classes.
 *
 * Lear/INRIA/Dalal's HOG binary file output implementations 
 * @modified by Feng Shi@discover lab, univ of ottawa @Apr 17, 2009
 */
// }}}

#ifndef _LEAR_BISTREAMABLE_H_
#define _LEAR_BISTREAMABLE_H_

// {{{ headers
#include <iostream>
// }}}


class BiIStream;
class BiOStream;

// {{{ class documentation
/**
 * @brief BiStreamable (Binary Streamable) class implements
 * "bload" and "bsave" interface for stremable objects.
 *
 * Any class, where we need to read and write objects in
 * binary format, must inherit BiStreamable class and 
 * implement public functions of BiStreamable class.
 *
 * This is Barton and Nackman inheritance trick. This way,
 * we do not need to define virtual functions. Compiler
 * will complain about missing functions at compile time.
 * 
 * You need to define your child class as something like this
 * \code
 * public MyClass : BiStreamable<MyClass> {

 * // your definition of MyClass
 *
 * // your implementation of public memebers of BiStreamable
 *
 * };
 * \endcode
 *
 */
// }}}
template<class T_leaftype >
class BiStreamable {
    protected:
        T_leaftype& leaf() {
            return static_cast<T_leaftype&> (*this);
        }
        const T_leaftype& leaf() const {
            return static_cast<const T_leaftype&> (*this);
        }
    public:
        typedef T_leaftype LeafType;

        inline BiIStream& bload(BiIStream& in) {
            leaf().bload(in);
            return in;
        }

        inline BiOStream& bsave(BiOStream& out) const {
            leaf().bsave(out);
            return out;
        }
};



#endif // _LEAR_BISTREAMABLE_H_

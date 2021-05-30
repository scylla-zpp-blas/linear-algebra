#pragma once

#include <algorithm>
#include <vector>

#include <scmd.hh>

#include "proto.hh"
#include "scylla_queue.hh"

#include "scylla_blas/matrix.hh"
#include "scylla_blas/vector.hh"

namespace scylla_blas::worker {

using procedure_t = std::optional<proto::response>(const std::shared_ptr<scmd::session>&, const proto::task&);

/* LEVEL 1 */
procedure_t sswap, sscal, scopy, saxpy, sdot, sdsdot, snrm2, sasum, isamax;
procedure_t dswap, dscal, dcopy, daxpy, ddot, dsdot, dnrm2, dasum, idamax;

/* LEVEL 2 */
procedure_t sgemv, sgbmv, strsv, stbsv, sger;
procedure_t dgemv, dgbmv, dtrsv, dtbsv, dger;

/* LEVEL 3 */
procedure_t sgemm, ssyrk, ssyr2k;
procedure_t dgemm, dsyrk, dsyr2k;

constexpr std::array<std::pair<proto::task_type, const procedure_t &>, 34> task_to_procedure =
{{
         {proto::SSWAP, sswap},
         {proto::SSCAL, sscal},
         {proto::SCOPY, scopy},
         {proto::SAXPY, saxpy},
         {proto::SDOT, sdot},
         {proto::SDSDOT, sdsdot},
         {proto::SNRM2, snrm2},
         {proto::SASUM, sasum},
         {proto::ISAMAX, isamax},

         {proto::DSWAP, dswap},
         {proto::DSCAL, dscal},
         {proto::DCOPY, dcopy},
         {proto::DAXPY, daxpy},
         {proto::DDOT, ddot},
         {proto::DSDOT, dsdot},
         {proto::DNRM2, dnrm2},
         {proto::DASUM, dasum},
         {proto::IDAMAX, idamax},

         {proto::SGEMV, sgemv},
         {proto::DGEMV, dgemv},
         {proto::SGBMV, sgemv},
         {proto::DGBMV, dgemv},
         {proto::STRSV, strsv},
         {proto::DTRSV, dtrsv},
         {proto::STBSV, stbsv},
         {proto::DTBSV, dtbsv},
         {proto::SGER, sger},
         {proto::DGER, dger},

         {proto::SGEMM, sgemm},
         {proto::DGEMM, dgemm},
         {proto::SSYRK, ssyrk},
         {proto::DSYRK, dsyrk},
         {proto::SSYR2K, ssyr2k},
         {proto::DSYR2K, dsyr2k}
 }};

constexpr procedure_t& get_procedure_for_task(const proto::task &t) {
    auto pred = [=](auto &val){ return val.first == t.type; };

    auto it = std::find_if(task_to_procedure.begin(), task_to_procedure.end(), pred);

    if (it == task_to_procedure.end()) {
        throw std::runtime_error("Operation type " + std::to_string(t.type) +  " not implemented!");
    }

    return it->second;
}

}
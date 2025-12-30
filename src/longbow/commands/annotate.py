import logging
import math
import re
import time
import sys
import os

import click
import click_log
import tqdm

import pysam

import longbow.utils
import longbow.utils.constants
from ..utils import bam_utils
from ..utils.bam_utils import collapse_annotations

logging.basicConfig(stream=sys.stderr)
logger = logging.getLogger("annotate")
click_log.basic_config(logger)


@click.command(name=logger.name)
@click_log.simple_verbosity_option(logger)
@click.option(
    "-p",
    "--pbi",
    required=False,
    type=click.Path(),
    help="BAM .pbi index file",
)
@click.option(
    "-t",
    "--threads",
    type=int,
    default=1,
    show_default=True,
    help="number of threads to use (JAX is multithreaded, using 1 to avoid deadlock)",
)
@click.option(
    "-o",
    "--output-bam",
    default="-",
    type=click.Path(exists=False),
    help="annotated bam output  [default: stdout]",
)
@click.option(
    "-m",
    "--model",
    type=str,
    help="The model(s) to use for annotation.  If the given value is a pre-configured model name, then that "
         "model will be used.  Otherwise, the given value will be treated as a file name and Longbow will attempt to "
         "read in the file and create a LibraryModel from it.  Longbow will assume the contents are the configuration "
         "of a LibraryModel as per LibraryModel.to_json()."
)
@click.option(
    "-c",
    "--chunk",
    type=str,
    default="",
    required=False,
    help="Process a single chunk of data (e.g. specify '2/4' to process the second of four equally-sized "
         "chunks across the dataset)"
)
@click.option(
    "-l",
    "--min-length",
    type=int,
    default=0,
    show_default=True,
    required=False,
    help="Minimum length of a read to process.  Reads shorter than this length will not be annotated."
)
@click.option(
    "-L",
    "--max-length",
    type=int,
    default=longbow.utils.constants.DEFAULT_MAX_READ_LENGTH,
    show_default=True,
    required=False,
    help="Maximum length of a read to process.  Reads longer than this length will not be annotated."
)
@click.option(
    "--min-rq",
    type=float,
    default=-2,
    show_default=True,
    required=False,
    help="Minimum ccs-determined read quality for a read to be annotated.  CCS read quality range is [-1,1]."
)
@click.option(
    '-f',
    '--force',
    is_flag=True,
    default=False,
    show_default=True,
    help="Force overwrite of the output files if they exist."
)
@click.argument("input-bam", default="-" if not sys.stdin.isatty() else None, type=click.File("rb"))
def main(pbi, threads, output_bam, model, chunk, min_length, max_length, min_rq, force, input_bam):
    """Annotate reads in a BAM file with segments from the model."""

    t_start = time.time()

    logger.info("Invoked via: longbow %s", " ".join(sys.argv[1:]))

    # Check to see if the output files exist:
    bam_utils.check_for_preexisting_files(output_bam, exist_ok=force)

    logger.info(f"Running with single-threaded mode (JAX is multithreaded, fork() would cause deadlock)")

    # Get our model:
    lb_model = bam_utils.load_model(model, input_bam)
    logger.info(f"Using {lb_model.name}: {lb_model.description}")

    pbi = f"{input_bam.name}.pbi" if pbi is None else pbi
    read_count = None
    read_num = 0
    start_offset = 0
    end_offset = math.inf

    if not os.path.exists(pbi) and chunk != "":
        raise ValueError(f"Chunking specified but pbi file '{pbi}' not found")

    if os.path.exists(pbi):
        if chunk != "":
            (chunk, num_chunks) = re.split("/", chunk)
            chunk = int(chunk)
            num_chunks = int(num_chunks)

            # Decode PacBio .pbi file and determine the shard offsets.
            offsets, zmw_counts, read_count, read_counts_per_chunk, read_nums = bam_utils.compute_shard_offsets(pbi, num_chunks)

            start_offset = offsets[chunk - 1]
            end_offset = offsets[chunk] if chunk < len(offsets) else offsets[chunk - 1]
            read_count = read_counts_per_chunk[chunk - 1] if chunk < len(offsets) else 0
            read_num = read_nums[chunk - 1] if chunk < len(offsets) else 0

            logger.info("Annotating %d reads from chunk %d/%d (reads %d-%d)", read_count, chunk, num_chunks, read_num, read_num + read_count - 1)
        else:
            read_count = bam_utils.load_read_count(pbi)
            logger.info("Annotating %d reads", read_count)
    else:
        read_count = bam_utils.get_read_count_from_bam_index(input_bam)
        if read_count:
            logger.info("Annotating %d reads", read_count)

    pysam.set_verbosity(0)  # silence message about the .bai file not being found
    with pysam.AlignmentFile(
        input_bam if start_offset == 0 else input_bam.name, "rb", check_sq=False, require_index=False
    ) as bam_file:

        # If we're chunking, advance to the specified virtual file offset.
        if start_offset > 0:
            bam_file.seek(start_offset)

        # Get our header from the input bam file:
        out_header_dict = bam_utils.create_bam_header_with_program_group(logger.name, bam_file.header, model=lb_model)
        out_header = pysam.AlignmentHeader.from_dict(out_header_dict)

        # Initialize counters
        num_reads_annotated = 0
        num_sections = 0

        with pysam.AlignmentFile(
            output_bam, "wb", header=out_header
        ) as out_bam_file, tqdm.tqdm(
            desc="Progress",
            unit=" read",
            colour="green",
            file=sys.stderr,
            disable=not sys.stdin.isatty(),
            total=read_count
        ) as pbar:

            for read in bam_file:
                # Check for chunking boundary
                if start_offset > 0 and bam_file.tell() >= end_offset:
                    break

                # Check for min/max length and min quality:
                if len(read.query_sequence) < min_length:
                    logger.debug(f"Read is shorter than min length.  "
                                 f"Skipping: {read.query_name} ({len(read.query_sequence)} < {min_length})")
                    pbar.update(1)
                    continue
                if len(read.query_sequence) > max_length:
                    logger.debug(f"Read is longer than max length.  "
                                 f"Skipping: {read.query_name} ({len(read.query_sequence)} > {max_length})")
                    pbar.update(1)
                    continue
                if read.get_tag("rq") < min_rq:
                    logger.debug(f"Read quality is below the minimum.  "
                                 f"Skipping: {read.query_name} ({read.get_tag('rq')} < {min_rq})")
                    pbar.update(1)
                    continue

                # Annotate the read
                read_str, best_path, best_logp, best_fit_is_rc = _annotate_and_assign_read_to_model(read, lb_model)

                if best_path is not None:
                    # Reconstruct the read object
                    read = pysam.AlignedSegment.fromstring(read_str, out_header)

                    # Obligatory log message:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "Path for read %s (%2.2f)%s: %s",
                            read.query_name,
                            best_logp,
                            " (RC)" if best_fit_is_rc else "",
                            ','.join(list(map(lambda x: f'{x.name}:{x.start}-{x.end}', collapse_annotations(best_path))))
                        )

                    # Write our our read:
                    bam_utils.write_annotated_read(read, best_path, best_fit_is_rc, best_logp, lb_model, out_bam_file)

                    # Increment our counters:
                    num_reads_annotated += 1
                    num_sections += len(best_path)

                pbar.update(1)

    logger.info(
        f"Annotated {num_reads_annotated} reads with {num_sections} total sections."
    )
    et = time.time()
    logger.info(f"Done. Elapsed time: {et - t_start:2.2f}s. "
                f"Overall processing rate: {num_reads_annotated/(et - t_start):2.2f} reads/s.")


def _annotate_and_assign_read_to_model(read, model):
    """Annotate the given read with the given model."""

    best_logp = -math.inf
    best_path = None
    best_fit_is_rc = False

    _, ppath, logp, is_rc = _annotate_read(read, model)

    if logp > best_logp:
        best_logp = logp
        best_path = ppath
        best_fit_is_rc = is_rc

    # Provide some info as to which model was chosen:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Sequence %s scored best %s." ,
            read.query_name,
            "in RC" if best_fit_is_rc else "in forward direction",
        )

    return read.to_string(), best_path, best_logp, best_fit_is_rc


def _annotate_read(read, model):
    is_rc = False
    logp, ppath = model.annotate(read.query_sequence)

    rc_logp, rc_ppath = model.annotate(bam_utils.reverse_complement(read.query_sequence))
    if rc_logp > logp:
        logp = rc_logp
        ppath = rc_ppath
        is_rc = True

    return read.to_string(), ppath, logp, is_rc

"""Comprehensive anonymisation test suite — Church abuse case materials.

Tests the full anonymisation pipeline against realistic litigation documents
for a church institutional abuse case. Designed to find edge cases in:
  - Victim/survivor identification
  - Relationship chains and indirect identifiers
  - Parish/school/institution naming
  - Published case citations vs current matter references
  - Privilege markers in advice letters
  - Settlement figures and quantum
  - Age/vulnerability categorisation
  - Medical/psychiatric terminology
  - Safeguarding-specific language
  - Multi-party references (clergy, church officials, social services)
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.anonymisation.service import AnonymisationService
from src.anonymisation.gateway import CloudExportGateway
from src.anonymisation.citation_guard import extract_protected_citations


# ---------------------------------------------------------------------------
# Test documents — realistic church abuse case materials
# ---------------------------------------------------------------------------

DOCS: dict[str, str] = {}

# ---- Document 1: Witness Statement (Survivor) ----
DOCS["witness_statement_survivor"] = """
WITNESS STATEMENT OF SARAH ELIZABETH THOMPSON

I, Sarah Elizabeth Thompson, of 47 Meadow Lane, Oakfield, Surrey GU21 5PQ, 
will say as follows:

1. I was born on 14th March 1982 and I am currently 43 years old. Between the 
ages of 8 and 12, I was a member of the St Augustine's Church choir in Oakfield, 
under the direction of Father Patrick Brennan.

2. The abuse began in approximately September 1990, when I was 8 years old and 
in Year 4 at Oakfield Church of England Primary School. Father Brennan would 
ask me to stay behind after choir practice on Wednesday evenings at the parish 
hall, 12 Church Lane, Oakfield.

3. On one occasion in or around November 1990, Father Brennan took me into the 
vestry of St Augustine's Church and sexually assaulted me. I was aged 8 at the 
time. I told my mother, Mrs Angela Thompson (née Richards), who reported the 
matter to the churchwarden, Mr Geoffrey Whitfield, in December 1990.

4. My mother has confirmed that Mr Whitfield told her he would "deal with it 
through the proper channels" but asked her not to involve the police as it would 
"cause terrible damage to the parish community."

5. I later learned from diocesan disclosure that Father Brennan had been the 
subject of a previous complaint by the family of another child, Thomas Williams, 
in 1987. The complaint was handled internally by Bishop Robert Caldwell of the 
Diocese of Guildford, who transferred Father Brennan from his previous parish 
in Woking to St Augustine's, Oakfield, without any safeguarding referral.

6. I reported the abuse to Surrey Police in March 2019 (crime reference 
CR/2019/0847632). Father Brennan was interviewed under caution but the CPS 
decided not to prosecute due to evidential difficulties.

7. I have suffered severe psychological consequences. Dr Helena Marsden, 
Consultant Psychiatrist at the Priory Hospital Woking, diagnosed me with 
complex PTSD and major depressive disorder in her report dated 15th June 2023. 
She attributes these conditions directly to the abuse.

8. I am now represented by Slater & Gordon Lawyers (ref: SG/CT/2023/4891) and 
bring this claim against the Diocese of Guildford. My solicitor is Ms Catherine 
Nguyen (SRA No. 612847).

STATEMENT OF TRUTH
I believe that the facts stated in this witness statement are true. I understand 
that proceedings for contempt of court may be brought against anyone who makes, 
or causes to be made, a false statement in a document verified by a statement 
of truth without an honest belief in its truth.

Signed: Sarah Elizabeth Thompson
Dated: 22nd September 2024
"""

# ---- Document 2: Solicitor-Client Advice Letter (PRIVILEGED) ----
DOCS["advice_letter_privileged"] = """
PRIVILEGED AND CONFIDENTIAL

Slater & Gordon Lawyers
Reference: SG/CT/2023/4891

22nd October 2024

Ms Sarah Thompson
47 Meadow Lane
Oakfield, Surrey
GU21 5PQ

Dear Ms Thompson,

Re: Thompson v The Diocese of Guildford — Claim for damages for institutional 
child sexual abuse

I write following our conference with counsel, Mr James Harrington-Scott QC, 
on 18th October 2024 at 2 Hare Court, Temple, London EC4Y 7BH.

Liability

The key legal issue is whether the Diocese of Guildford is vicariously liable 
for the acts of Father Brennan. Following the Supreme Court's decision in 
The Catholic Child Welfare Society v Various Claimants [2012] UKSC 56 
(the "Christian Brothers" case), the test is whether the relationship between 
the tortfeasor and the defendant is sufficiently akin to employment, and whether 
there is a sufficient connection between that relationship and the tortfeasor's 
wrongdoing.

In Maga v Archbishop of Birmingham [2010] EWCA Civ 256, the Court of Appeal 
held that a diocese could be vicariously liable for the sexual abuse committed 
by a priest where the priest's role gave him the opportunity to commit the abuse. 
This principle was further developed in JGE v The Trustees of the Portsmouth 
Roman Catholic Diocesan Trust [2012] EWCA Civ 938.

The evidence that Father Brennan abused his position of trust as parish priest 
and choir director, combined with the Diocese's failure to act on the 1987 
Williams complaint, establishes a strong case on liability.

Quantum

Counsel's preliminary view on quantum, based on the Judicial College Guidelines 
(16th edition, 2024) and comparable awards:

- General damages for PTSD (severe): £59,860 to £100,670
- General damages for sexual abuse (Category A — serious): £54,000 to £115,000
- Special damages (therapy costs to date): approximately £28,400
- Future therapy costs (estimated 5 years): approximately £45,000
- Loss of earnings differential: approximately £180,000

Total estimated claim value: £367,260 to £469,070

The defendant has offered GBP 85,000 in a without prejudice Part 36 offer 
dated 15th October 2024, which counsel considers substantially below a 
reasonable settlement range.

Next Steps

We recommend rejecting the Part 36 offer and proceeding to serve a Schedule 
of Loss within 21 days. Counsel has also advised that we obtain a further 
psychiatric report from an expert instructed jointly with the defendant.

Please contact me on 01234 567890 or catherine.nguyen@slatergordon.co.uk 
to discuss.

Yours sincerely,

Catherine Nguyen
Partner, Abuse Claims
Slater & Gordon Lawyers
SRA No. 612847
"""

# ---- Document 3: Expert Psychiatric Report ----
DOCS["psychiatric_report"] = """
EXPERT PSYCHIATRIC REPORT

Prepared for: Slater & Gordon Lawyers (ref: SG/CT/2023/4891)

Patient: Ms Sarah Elizabeth Thompson (DOB: 14/03/1982)
Date of examination: 10th June 2023
Report date: 15th June 2023
Expert: Dr Helena Marsden, MB BS, FRCPsych, MPhil
         Consultant Psychiatrist
         The Priory Hospital Woking
         Redwood House, Chertsey Road, Woking GU21 8HL
         GMC No. 4578123

INSTRUCTIONS

I was instructed by Ms Catherine Nguyen of Slater & Gordon Lawyers on behalf 
of the Claimant to prepare a psychiatric report addressing the psychological 
consequences of alleged sexual abuse by Father Patrick Brennan between 
approximately 1990 and 1994, when the Claimant was aged between 8 and 12 years.

HISTORY

Ms Thompson reported that the abuse began in September 1990 when she was in 
Year 4 at Oakfield Church of England Primary School. She described incidents 
of sexual touching and more serious sexual assault occurring in the vestry of 
St Augustine's Church, Oakfield, and at the parish hall at 12 Church Lane.

She reported the abuse to her mother, Mrs Angela Thompson, in late 1990. Her 
mother raised the matter with the churchwarden, Geoffrey Whitfield, but no 
action was taken by the church authorities. Ms Thompson did not disclose the 
abuse again until 2019, when she reported to Surrey Police.

Ms Thompson described significant psychological symptoms beginning in 
adolescence, including:
- Nightmares and flashbacks (from approximately age 13)
- Self-harm (cutting, aged 15-17)
- Eating disorder (anorexia nervosa, aged 16, treated at the Maudsley Hospital)
- Cannabis dependence (aged 18-25)
- Two episodes of suicidal ideation (2005 and 2018)
- Relationship difficulties, including the breakdown of her marriage to 
  Mr David Thompson in 2020

CURRENT MEDICATION
Sertraline 150mg daily (prescribed by Dr Amara Osei, Oakfield Medical Centre, 
NHS No. 4832918472)

DIAGNOSIS

Based on my clinical assessment and review of the medical records, I diagnose:

1. Complex Post-Traumatic Stress Disorder (ICD-11: 6B41)
   - Directly attributable to the sexual abuse by Father Brennan
   
2. Major Depressive Disorder, recurrent, moderate (ICD-11: 6A71)
   - Substantially contributed to by the abuse and its sequelae

3. Disordered eating (in remission)
   - Historical, causally linked to the abuse

PROGNOSIS

Without further treatment, Ms Thompson's conditions are likely to persist 
indefinitely. With appropriate trauma-focused therapy (EMDR or trauma-focused 
CBT), I would expect improvement over 3-5 years, although complete resolution 
is unlikely given the severity and duration of the abuse and the institutional 
betrayal.

I estimate the cost of appropriate treatment at approximately £180 per session, 
fortnightly, for 5 years (approximately 130 sessions = £23,400).

STATEMENT OF TRUTH
I confirm that I have made clear which facts and matters referred to in this 
report are within my own knowledge and which are not. Those that are within my 
own knowledge I confirm to be true. The opinions I have expressed represent my 
true and complete professional opinions on the matters to which they refer.

Dr Helena Marsden
15th June 2023
"""

# ---- Document 4: Safeguarding Report (Internal Church Document) ----
DOCS["safeguarding_report"] = """
DIOCESE OF GUILDFORD
SAFEGUARDING FILE — CONFIDENTIAL

Reference: SF/1990/0023
Date opened: 15th December 1990
Date closed: 28th February 1991

Subject: Complaint regarding the Reverend Patrick James Brennan
Parish: St Augustine's, Oakfield
Complainant: Mrs Angela Thompson, 47 Meadow Lane, Oakfield GU21 5PQ
             (mother of the alleged victim)

1. COMPLAINT RECEIVED

On 14th December 1990, the Churchwarden of St Augustine's, Mr Geoffrey 
Whitfield, informed the Diocesan Secretary that he had received a verbal 
complaint from Mrs Angela Thompson alleging inappropriate behaviour by 
Fr Brennan towards her daughter, Sarah (aged 8).

Mrs Thompson alleged that Fr Brennan had kept her daughter behind after 
choir practice and behaved inappropriately towards her in the church vestry.

2. PREVIOUS COMPLAINTS

It is noted that Fr Brennan was the subject of a complaint in 1987 whilst 
serving at Holy Trinity Church, Woking. The complaint was made by Mr and 
Mrs Raymond Williams regarding their son Thomas (then aged 10). Bishop 
Robert Caldwell dealt with the matter personally and arranged Fr Brennan's 
transfer to St Augustine's, Oakfield, effective from Easter 1988.

No referral was made to social services or the police regarding the 1987 
complaint. The Williams family was asked to keep the matter confidential 
"for the sake of the parish."

3. ACTION TAKEN

The Diocesan Secretary, Canon Michael Hurst, discussed the matter with 
Bishop Caldwell on 16th December 1990. Bishop Caldwell decided that:

(a) Fr Brennan would be spoken to privately and asked to "exercise greater 
    caution in his interactions with young parishioners"
(b) No referral would be made to Surrey Social Services or the police
(c) Mrs Thompson would be asked to accept the church's assurance that 
    the matter would be "dealt with pastorally"
(d) The complaint would not be recorded in Fr Brennan's personnel file

4. OUTCOME

Canon Hurst wrote to Mrs Thompson on 28th February 1991 confirming that 
"appropriate pastoral measures have been taken" and that the Diocese 
"considers the matter closed."

Fr Brennan continued to serve at St Augustine's until his retirement in 2010.

Filed by: Canon Michael Hurst, Diocesan Secretary
Countersigned: Bishop Robert Caldwell
"""

# ---- Document 5: Without Prejudice Settlement Correspondence ----
DOCS["settlement_wp"] = """
WITHOUT PREJUDICE
SAVE AS TO COSTS

Kennedys Law
Reference: KEN/DG/2024/7732

25th October 2024

Slater & Gordon Lawyers
For the attention of Ms Catherine Nguyen
Reference: SG/CT/2023/4891

Dear Ms Nguyen,

Re: Thompson v The Diocese of Guildford — Without Prejudice

We write on behalf of our client, the Diocese of Guildford, with an updated 
settlement proposal.

Our client has carefully considered the medical evidence of Dr Helena Marsden 
and the updated Schedule of Loss served on 20th October 2024.

Without any admission of liability, and noting that our client continues to 
deny the allegations of Father Brennan's conduct as described in the 
Particulars of Claim, our client is prepared to increase its offer from 
GBP 85,000 to GBP 175,000 inclusive of costs, in full and final settlement 
of all claims.

This offer is made as a Part 36 offer pursuant to CPR Part 36 and is open 
for acceptance for 21 days from the date of this letter.

In making this offer, our client takes into account:

1. The evidential difficulties arising from the passage of time (the alleged 
   events occurred 34 years ago)
2. The absence of any criminal conviction or finding of fact against Fr Brennan
3. The limitation issues arising from Limitation Act 1980, section 11, noting 
   that the claim was not brought within three years of the Claimant's 18th 
   birthday, and reliance must be placed on the court's discretion under 
   section 33
4. The decision in A v Hoare [2008] UKHL 6 which held that the limitation 
   period for intentional trespass to the person runs from the date of accrual, 
   and the court's broad discretion under section 33

Our client also takes account of the principles in Raggett v Society of 
Jesus Trust 1929 [2010] EWCA Civ 1002 regarding the assessment of damages 
in historical abuse cases, and the more recent guidance in DSN v Blackpool 
Football Club Ltd [2021] EWCA Civ 1352 on the standard of proof in sexual 
abuse claims.

We strongly urge your client to accept this offer. Should the matter proceed 
to trial, our client will rely on this offer on the question of costs.

Yours faithfully,

Michael Chen
Partner, Insurance Litigation
Kennedys Law
Direct line: 020 7667 9345
michael.chen@kennedys-law.com
"""

# ---- Document 6: Internal Church Correspondence (Cover-up) ----
DOCS["church_internal_memo"] = """
STRICTLY PRIVATE AND CONFIDENTIAL
For the personal attention of the Bishop only

From: Canon Michael Hurst, Diocesan Secretary
To: The Right Reverend Robert Caldwell, Bishop of Guildford
Date: 16th December 1990

My Lord Bishop,

I write regarding the Thompson complaint about Fr Brennan.

As you will recall, this is the second complaint of this nature concerning 
Fr Brennan. The Williams complaint in 1987 was dealt with by Your Lordship 
personally, and Fr Brennan was moved from Holy Trinity, Woking, to 
St Augustine's, Oakfield.

I have spoken with the churchwarden, Mr Geoffrey Whitfield, who tells me 
that Mrs Angela Thompson is "very upset but not the sort to make a fuss." 
He believes the matter can be contained within the parish.

However, I am concerned that if a further complaint were to be made — 
particularly to the police or social services — Fr Brennan's previous 
history could come to light, which would create very serious difficulties 
for the Diocese.

I would recommend that we:

1. Arrange for Fr Brennan to take a period of retreat at Quarr Abbey, 
   Isle of Wight, for three months from January 1991
2. Ensure that Mrs Thompson receives pastoral support from the curate, 
   Fr David Mitchell, to maintain good relations
3. Consider whether Fr Brennan should be moved to a parish without a 
   primary school

I would be grateful for Your Lordship's guidance.

Yours faithfully in Christ,

Canon Michael Hurst
Diocesan Secretary

P.S. I have arranged for the safeguarding file to be held separately from 
Fr Brennan's main personnel file, as discussed.
"""

# ---- Document 7: Edge case — Dense cross-references ----
DOCS["edge_case_dense_refs"] = """
TIMELINE OF KEY EVENTS — Thompson v Diocese of Guildford

September 1987: Thomas Williams (aged 10) abused by Fr Brennan at Holy 
Trinity, Woking. Complaint by Raymond & Mary Williams to Bishop Caldwell.

Easter 1988: Fr Brennan transferred from Holy Trinity, Woking, to 
St Augustine's, Oakfield. No safeguarding referral. Williams family asked 
to keep silent.

September 1990: Sarah Thompson (aged 8) joins St Augustine's choir.

November 1990: First incident of sexual assault on Sarah Thompson in the 
vestry of St Augustine's Church by Fr Brennan.

December 1990: Angela Thompson (Sarah's mother) reports to churchwarden 
Geoffrey Whitfield. Whitfield reports to Diocesan Secretary Canon Hurst. 
Bishop Caldwell decides: no police, no social services, pastoral handling.

February 1991: Canon Hurst writes to Angela Thompson: "matter closed."

1994: Abuse of Sarah Thompson ends when Fr Brennan is moved to 
St Bartholomew's, Farnham. Fr Brennan retires 2010.

March 2019: Sarah Thompson reports to Surrey Police. Crime ref CR/2019/0847632.

June 2023: Dr Helena Marsden (Priory Hospital Woking, GMC 4578123) diagnoses 
Sarah with complex PTSD.

October 2024: Diocese offers GBP 85,000 Part 36 (rejected). Updated offer 
GBP 175,000.

Key contacts: Ms Catherine Nguyen, Slater & Gordon (SRA 612847). 
Mr Michael Chen, Kennedys Law. Mr James Harrington-Scott QC, 2 Hare Court.
Dr Amara Osei, Oakfield Medical Centre. NHS No. 4832918472.
"""

# ---- Document 8: Edge case — Ages and vulnerability ----
DOCS["edge_case_ages"] = """
SCHEDULE OF VICTIMS — Diocese of Guildford Safeguarding Review 2023

Ref     Name               Age at      Age Now  Parish              Perpetrator
                           Abuse
VS-001  Sarah Thompson     8-12        43       St Augustine's      Fr Brennan
VS-002  Thomas Williams    10          47       Holy Trinity         Fr Brennan
VS-003  Emily Carter       6           38       St Bartholomew's    Fr Brennan
VS-004  James Okonkwo      14          35       St Augustine's      Fr Mitchell
VS-005  Lucy Patel-Shah    7           29       Holy Trinity         Fr Brennan
VS-006  BABY A             2           18       St Augustine's       Fr Brennan
        (toddler at time)
VS-007  Michael Nowak      15          31       St Bartholomew's    Fr Brennan
VS-008  Grace O'Sullivan   87          92       Care of Fr Brennan   Fr Brennan
        (vulnerable adult)
VS-009  Margaret Chen-Liu  78          83       St Augustine's       Fr Mitchell
        (vulnerable adult, dementia)

Note: VS-006 is a particularly sensitive case involving a very young child. 
The family has requested maximum anonymity. Baby A's real name is 
Olivia Rose Henderson, daughter of Mr and Mrs Robert Henderson, 
14 Primrose Close, Oakfield GU21 7TH.

VS-008 and VS-009 involve elderly vulnerable adults where the abuse was 
non-sexual but constituted financial exploitation and emotional manipulation.
Grace O'Sullivan, aged 87, was induced to change her will in favour of 
Fr Brennan. Margaret Chen-Liu, aged 78 and suffering from early-stage 
dementia, was financially exploited for approximately GBP 340,000.
"""

# ---- Document 9: Edge case — Statutory references and legal analysis ----
DOCS["edge_case_legal_analysis"] = """
NOTE ON LIMITATION — Thompson v Diocese of Guildford

Prepared by Mr James Harrington-Scott QC

1. The primary limitation period for personal injury claims is three years 
from the date of accrual (or the date of knowledge): Limitation Act 1980, 
section 11(4). For a claimant who was a minor at the time of the cause of 
action, time does not begin to run until the claimant's 18th birthday: 
section 28(1).

2. Ms Thompson's 18th birthday was 14th March 2000. The primary limitation 
period therefore expired on 14th March 2003. The claim was not issued until 
November 2023, some 20 years out of time.

3. However, section 33 of the Limitation Act 1980 confers a broad discretion 
on the court to disapply the limitation period where it would be equitable 
to do so. The court must have regard to the factors in section 33(3)(a)-(f).

4. In A v Hoare [2008] UKHL 6, the House of Lords held (per Lord Hoffmann 
at [49]) that the section 33 discretion should be exercised "unfettered" in 
sexual abuse cases and that the passage of time alone is not a sufficient 
reason to refuse to disapply.

5. The key question under section 33 is whether a fair trial is still 
possible: KR v Bryn Alyn Community Holdings Ltd [2003] EWCA Civ 85. In 
Catholic Care (Diocese of Leeds) v Young [2006] EWCA Civ 1534, the Court 
of Appeal held that the institutional defendant's own failure to keep records 
should not be held against the claimant.

6. More recently, in FZO v Adams [2024] EWCA Civ 1111, the Court of Appeal 
confirmed that in cases of institutional child sexual abuse, there is a 
strong presumption in favour of disapplying the limitation period unless the 
defendant can demonstrate real and substantial prejudice beyond the mere 
passage of time.

7. I advise that we have strong prospects of obtaining a section 33 
disapplication in this case, particularly given:
(a) The Diocese's own destruction of safeguarding records
(b) The active concealment of the 1987 Williams complaint
(c) Fr Brennan's continued access to children until 2010
(d) The inherent difficulty for child abuse survivors in making timely 
    complaints (see B v Nugent Care Society [2009] EWCA Civ 827)
"""


# ===========================================================================
# Test harness
# ===========================================================================

@dataclass
class TestResult:
    """Result of a single anonymisation test."""

    doc_name: str
    entity_count: int
    privilege_markers: int
    citations_preserved: int
    validation_passed: bool
    edge_cases: list[str]
    sample_output: str  # First 500 chars of anonymised text


def run_test(
    service: AnonymisationService,
    gateway: CloudExportGateway,
    doc_name: str,
    text: str,
) -> TestResult:
    """Run the full anonymisation pipeline on a single test document."""
    
    edge_cases: list[str] = []
    
    # Export through gateway
    payload = gateway.export_document(
        text=text,
        source_document_id=doc_name,
        source_filename=f"{doc_name}.pdf",
    )
    
    anon_text = payload.anonymised_text
    
    # --- Edge case checks ---
    
    # 1. Check for leaked real names (should NOT appear in output)
    leak_names = [
        # Full names
        "Sarah Thompson", "Sarah Elizabeth Thompson", "Angela Thompson",
        "Patrick Brennan", "Geoffrey Whitfield", "Robert Caldwell",
        "Michael Hurst", "Thomas Williams", "Raymond Williams",
        "Catherine Nguyen", "James Harrington-Scott", "Helena Marsden",
        "David Thompson", "Amara Osei", "Michael Chen", "David Mitchell",
        "Emily Carter", "James Okonkwo", "Lucy Patel-Shah",
        "Olivia Rose Henderson", "Robert Henderson", "Grace O'Sullivan",
        "Margaret Chen-Liu", "Michael Nowak", "Mary Williams",
    ]
    # Also check standalone surnames/identifying words
    import re
    leak_surnames = [
        "Thompson", "Brennan", "Whitfield", "Caldwell", "Hurst",
        "Nguyen", "Marsden", "Osei", "Henderson", "Nowak",
        "Patel-Shah", "Okonkwo", "O'Sullivan", "Chen-Liu",
    ]
    for name in leak_surnames:
        # Only flag if it appears outside of token brackets
        pattern = re.compile(
            r"(?<!\[)\b" + re.escape(name) + r"\b(?![^\[]*\])",
            re.IGNORECASE,
        )
        if pattern.search(anon_text):
            edge_cases.append(f"SURNAME LEAK: '{name}' survived anonymisation")
    for name in leak_names:
        if name in anon_text:
            edge_cases.append(f"NAME LEAK: '{name}' survived anonymisation")
    
    # 2. Check for leaked addresses
    leak_addresses = [
        "47 Meadow Lane", "12 Church Lane", "14 Primrose Close",
        "GU21 5PQ", "GU21 7TH", "GU21 8HL", "EC4Y 7BH",
    ]
    for addr in leak_addresses:
        if addr in anon_text:
            edge_cases.append(f"ADDRESS LEAK: '{addr}' survived anonymisation")
    
    # 3. Check for leaked contact details
    leak_contacts = [
        "01234 567890", "catherine.nguyen@slatergordon.co.uk",
        "020 7667 9345", "michael.chen@kennedys-law.com",
        "612847",  # SRA number
        "4578123",  # GMC number
        "4832918472",  # NHS number
        "CR/2019/0847632",  # Crime reference
    ]
    for contact in leak_contacts:
        if contact in anon_text:
            edge_cases.append(f"CONTACT LEAK: '{contact}' survived anonymisation")
    
    # 4. Check that published case citations are PRESERVED
    expected_citations = []
    if "Hoare" in text:
        expected_citations.append("[2008] UKHL 6")
    if "Christian Brothers" in text or "Catholic Child Welfare" in text:
        expected_citations.append("[2012] UKSC 56")
    if "Maga" in text:
        expected_citations.append("[2010] EWCA Civ 256")
    if "JGE" in text:
        expected_citations.append("[2012] EWCA Civ 938")
    if "Raggett" in text:
        expected_citations.append("[2010] EWCA Civ 1002")
    if "DSN" in text:
        expected_citations.append("[2021] EWCA Civ 1352")
    if "FZO" in text:
        expected_citations.append("[2024] EWCA Civ 1111")
    if "KR v Bryn" in text:
        expected_citations.append("[2003] EWCA Civ 85")
    if "Catholic Care" in text:
        expected_citations.append("[2006] EWCA Civ 1534")
    if "Nugent" in text:
        expected_citations.append("[2009] EWCA Civ 827")
    
    for cit in expected_citations:
        if cit not in anon_text:
            edge_cases.append(f"CITATION LOST: '{cit}' was anonymised (should be preserved)")
    
    # 5. Check that contextual tokens have roles/magnitudes
    import re
    person_tokens = re.findall(r"\[PERSON_NAME_\d+(?::(\w+))?\]", anon_text)
    person_tokens_with_role = [t for t in person_tokens if t]
    if person_tokens and not person_tokens_with_role:
        edge_cases.append("NO ROLES: Person tokens lack role context (e.g. :director, :client)")
    
    amount_tokens = re.findall(r"\[MONETARY_AMOUNT_\d+(?::(\w+))?\]", anon_text)
    amount_tokens_with_mag = [t for t in amount_tokens if t]
    if amount_tokens and not amount_tokens_with_mag:
        edge_cases.append("NO MAGNITUDE: Monetary tokens lack magnitude (e.g. :six_figures)")
    
    # 6. Check for mangled adjacent tokens
    if "][" in anon_text and "] [" not in anon_text:
        # Find the specific mangling
        mangles = re.findall(r"\]\[", anon_text)
        if mangles:
            edge_cases.append(f"TOKEN MANGLING: {len(mangles)} adjacent tokens without spacing")
    
    # 7. Check that dates preserve years where possible
    date_tokens = re.findall(r"\[DATE_\d+(?::(\d{4}))?\]", anon_text)
    date_tokens_with_year = [t for t in date_tokens if t]
    if date_tokens and not date_tokens_with_year:
        edge_cases.append("NO YEARS: Date tokens lack year context (e.g. :2024)")
    
    # 8. Check for specific abuse-related identifiers that must be caught
    if "St Augustine's" in anon_text and "St Augustine's Church" not in text[:50]:
        # St Augustine's is the parish — it's an institution identifier
        # (We don't flag this as a hard failure since it's a church name,
        # but note it for review)
        edge_cases.append("NOTE: Parish name 'St Augustine\\'s' survives — review if identifying")
    
    # Count preserved citations
    protected = extract_protected_citations(text)
    
    return TestResult(
        doc_name=doc_name,
        entity_count=payload.metadata.get("entity_count", 0),
        privilege_markers=payload.metadata.get("privilege_markers_found", 0),
        citations_preserved=len(protected),
        validation_passed=payload.validation_passed,
        edge_cases=edge_cases,
        sample_output=anon_text[:800],
    )


def main() -> None:
    """Run the full test suite."""
    
    tmpdir = tempfile.mkdtemp()
    
    service = AnonymisationService(
        matter_id="thompson_v_guildford",
        passphrase="test_passphrase_2024",
        db_path=os.path.join(tmpdir, "test_anon.db"),
        use_presidio=False,  # Disabled on ROCm
        use_spacy=True,
        use_patterns=True,
        use_llm=False,
        double_pass_validation=False,
    )
    
    gateway = CloudExportGateway(
        service=service,
        block_on_validation_failure=False,
        block_on_pending_review=False,
        target_provider="anthropic",  # Claude Teams
        privilege_mode="anonymise",
    )
    
    print("=" * 78)
    print("PROJECT XX — ANONYMISATION TEST SUITE")
    print("Church Abuse Case Materials (Thompson v Diocese of Guildford)")
    print(f"Target provider: Anthropic Claude Teams (US-based, no ZDR)")
    print("=" * 78)
    print()
    
    results: list[TestResult] = []
    total_edge_cases: list[str] = []
    
    for doc_name, text in DOCS.items():
        print(f"Testing: {doc_name}...")
        result = run_test(service, gateway, doc_name, text)
        results.append(result)
        
        status = "PASS" if not result.edge_cases else f"ISSUES ({len(result.edge_cases)})"
        print(f"  Entities: {result.entity_count}, "
              f"Privilege: {result.privilege_markers}, "
              f"Citations preserved: {result.citations_preserved}, "
              f"Status: {status}")
        
        if result.edge_cases:
            for ec in result.edge_cases:
                print(f"    >> {ec}")
                total_edge_cases.append(f"[{doc_name}] {ec}")
        print()
    
    # Summary
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    total_entities = sum(r.entity_count for r in results)
    total_privilege = sum(r.privilege_markers for r in results)
    total_citations = sum(r.citations_preserved for r in results)
    docs_with_issues = sum(1 for r in results if r.edge_cases)
    
    print(f"Documents tested:      {len(results)}")
    print(f"Total entities found:  {total_entities}")
    print(f"Privilege markers:     {total_privilege}")
    print(f"Citations preserved:   {total_citations}")
    print(f"Documents with issues: {docs_with_issues}/{len(results)}")
    print(f"Total edge cases:      {len(total_edge_cases)}")
    print()
    
    if total_edge_cases:
        print("EDGE CASES REQUIRING ATTENTION:")
        print("-" * 40)
        
        # Categorise
        leaks = [e for e in total_edge_cases if "LEAK" in e]
        citation_issues = [e for e in total_edge_cases if "CITATION" in e]
        token_issues = [e for e in total_edge_cases if "TOKEN" in e or "NO " in e.split("] ", 1)[-1]]
        notes = [e for e in total_edge_cases if "NOTE:" in e]
        
        if leaks:
            print(f"\n  CRITICAL — Data Leaks ({len(leaks)}):")
            for e in leaks:
                print(f"    {e}")
        
        if citation_issues:
            print(f"\n  HIGH — Citation Issues ({len(citation_issues)}):")
            for e in citation_issues:
                print(f"    {e}")
        
        if token_issues:
            print(f"\n  MEDIUM — Token Quality ({len(token_issues)}):")
            for e in token_issues:
                print(f"    {e}")
        
        if notes:
            print(f"\n  LOW — Notes ({len(notes)}):")
            for e in notes:
                print(f"    {e}")
    else:
        print("ALL TESTS PASSED — No edge cases found.")
    
    # Print sample output for the most complex document
    print()
    print("=" * 78)
    print("SAMPLE ANONYMISED OUTPUT — advice_letter_privileged")
    print("=" * 78)
    for r in results:
        if r.doc_name == "advice_letter_privileged":
            print(r.sample_output)
            print("...")
            break
    
    print()
    print("=" * 78)
    print("SAMPLE ANONYMISED OUTPUT — safeguarding_report")
    print("=" * 78)
    for r in results:
        if r.doc_name == "safeguarding_report":
            print(r.sample_output)
            print("...")
            break
    
    service.close()
    print("\nDone.")


if __name__ == "__main__":
    main()

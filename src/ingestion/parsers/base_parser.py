"""Base parser interface and utilities."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.schema import DocumentType, ParsedDocument


@dataclass
class ClassificationResult:
    """Result of document classification with confidence."""
    document_type: DocumentType
    confidence: float  # 0.0 to 1.0
    matched_patterns: list[str]  # Patterns that matched


class BaseParser(ABC):
    """Abstract base class for document parsers.

    All parsers must implement the parse method to extract text and metadata
    from documents. Parsers should handle errors gracefully and log failures.
    """

    @abstractmethod
    def parse(self, file_path: str | Path, document_type: Optional[DocumentType] = None) -> ParsedDocument:
        """Parse a document and extract text and metadata.

        Args:
            file_path: Path to the document file
            document_type: Optional document type override. If None, parser
                          should attempt to detect the type.

        Returns:
            ParsedDocument with extracted text and metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
            Exception: Other parsing errors should be caught and logged
        """
        pass

    @abstractmethod
    def can_parse(self, file_path: str | Path) -> bool:
        """Check if this parser can handle the given file.

        Args:
            file_path: Path to the file

        Returns:
            True if parser can handle this file type
        """
        pass

    def detect_document_type(self, file_path: str | Path, text: str, metadata: dict) -> DocumentType:
        """Detect document type from file path, text, and metadata.

        Uses a comprehensive weighted scoring system optimized for UK litigation
        documents (civil fraud & competition law).

        Args:
            file_path: Path to the file
            text: Extracted text content
            metadata: Parsed metadata

        Returns:
            Detected document type
        """
        result = self.classify_document(file_path, text, metadata)
        return result.document_type

    def classify_document(self, file_path: str | Path, text: str, metadata: dict) -> ClassificationResult:
        """Classify document with confidence score and matched patterns.

        Args:
            file_path: Path to the file
            text: Extracted text content
            metadata: Parsed metadata

        Returns:
            ClassificationResult with type, confidence, and matched patterns
        """
        file_path_str = str(file_path).lower()
        file_name = Path(file_path).stem.lower()
        text_lower = text.lower()
        text_start = text_lower[:3000]  # First 3000 chars for header detection
        text_first_page = text_lower[:5000]  # Approximate first page
        
        # Track matched patterns for debugging
        matched_patterns: dict[str, list[str]] = {doc_type: [] for doc_type in self._get_all_types()}
        
        # Score each document type
        scores: dict[str, float] = {doc_type: 0.0 for doc_type in self._get_all_types()}
        
        # =====================================================================
        # WITNESS STATEMENT
        # =====================================================================
        ws_patterns = [
            (r"witness\s*statement", text_start, 6, "witness statement header"),
            (r"i\s+make\s+this\s+statement", text_start, 5, "I make this statement"),
            (r"statement\s+of\s+truth", text_lower, 5, "statement of truth"),
            (r"i\s+believe\s+(?:that\s+)?the\s+facts", text_lower, 4, "I believe the facts"),
            (r"true\s+to\s+the\s+best\s+of\s+my\s+knowledge", text_lower, 4, "true to best of knowledge"),
            (r"my\s+name\s+is\s+[a-z]+", text_start, 3, "my name is..."),
            (r"i\s+am\s+(?:employed|a\s+director|the\s+claimant)", text_start, 3, "I am employed/director/claimant"),
            (r"ws[-_]?\d+|witness[-_]", file_name, 4, "filename pattern WS"),
        ]
        for pattern, search_text, weight, desc in ws_patterns:
            if re.search(pattern, search_text):
                scores["witness_statement"] += weight
                matched_patterns["witness_statement"].append(desc)
        
        # First-person narrative density
        first_person = len(re.findall(r"\bi\s+(?:am|was|have|had|will|would|believe|saw|heard|know|met|spoke)", text_start))
        if first_person > 8:
            scores["witness_statement"] += 3
            matched_patterns["witness_statement"].append(f"first-person density ({first_person})")

        # =====================================================================
        # PLEADING (Claims, Defences, Replies, Counterclaims)
        # =====================================================================
        pleading_patterns = [
            (r"particulars\s+of\s+claim", text_start, 7, "particulars of claim"),
            (r"defence\s+(?:and\s+counterclaim)?", text_start, 6, "defence header"),
            (r"reply\s+(?:to\s+)?defence", text_start, 6, "reply to defence"),
            (r"counterclaim", text_start, 5, "counterclaim"),
            (r"in\s+the\s+(?:high|county|commercial)\s+court", text_start, 5, "court header"),
            (r"claim\s*(?:no|number)[:\s]*\w+", text_start, 4, "claim number"),
            (r"between[:\s]*\n", text_start, 3, "between parties"),
            (r"(?:claimant|defendant|respondent|applicant)s?[:\s]*\n", text_start, 3, "party labels"),
            (r"(?:poc|defence|reply)[-_]", file_name, 4, "filename pattern"),
            (r"(?:breach\s+of\s+)?(?:contract|duty|fiduciary)", text_lower, 2, "legal claims"),
            (r"(?:fraud|deceit|misrepresentation|conspiracy)", text_lower, 3, "fraud claims"),
            (r"(?:unlawful\s+means|economic\s+tort)", text_lower, 3, "tort claims"),
        ]
        for pattern, search_text, weight, desc in pleading_patterns:
            if re.search(pattern, search_text):
                scores["pleading"] += weight
                matched_patterns["pleading"].append(desc)

        # Numbered paragraphs typical of pleadings
        numbered_paras = len(re.findall(r"^\s*\d+\.\s+", text_lower, re.MULTILINE))
        if numbered_paras > 10:
            scores["pleading"] += 2
            matched_patterns["pleading"].append(f"numbered paragraphs ({numbered_paras})")

        # =====================================================================
        # SKELETON ARGUMENT
        # =====================================================================
        skeleton_patterns = [
            (r"skeleton\s+argument", text_start, 8, "skeleton argument header"),
            (r"submissions?\s+(?:on\s+behalf\s+of|for\s+the)", text_start, 6, "submissions header"),
            (r"written\s+submissions?", text_start, 5, "written submissions"),
            (r"(?:counsel|solicitors?)\s+for\s+the\s+(?:claimant|defendant)", text_start, 4, "counsel for"),
            (r"skeleton[-_]", file_name, 5, "filename pattern"),
            (r"it\s+is\s+(?:submitted|contended)", text_lower, 4, "it is submitted"),
            (r"the\s+(?:claimant|defendant)\s+(?:submits|contends)", text_lower, 3, "party submits"),
            (r"(?:ground|issue)\s+\d+", text_lower, 2, "numbered grounds/issues"),
        ]
        for pattern, search_text, weight, desc in skeleton_patterns:
            if re.search(pattern, search_text):
                scores["skeleton_argument"] += weight
                matched_patterns["skeleton_argument"].append(desc)

        # =====================================================================
        # EXPERT REPORT
        # =====================================================================
        expert_patterns = [
            (r"expert\s+(?:report|opinion|evidence)", text_start, 7, "expert report header"),
            (r"(?:forensic|accounting|financial|medical|psychiatric|engineering)\s+expert", text_start, 5, "expert type"),
            (r"cpr\s+35|part\s+35", text_lower, 6, "CPR Part 35 reference"),
            (r"instructions?\s+to\s+(?:the\s+)?expert", text_lower, 4, "instructions to expert"),
            (r"expert[-_]|report[-_]expert", file_name, 4, "filename pattern"),
            (r"my\s+(?:qualifications?|expertise|experience)", text_start, 4, "expert qualifications"),
            (r"(?:jointly|single|joint)\s+instructed", text_lower, 4, "joint instruction"),
            (r"i\s+(?:am|have\s+been)\s+(?:instructed|asked)", text_start, 3, "I am instructed"),
            (r"(?:appendix|annex|schedule)\s+[a-z0-9]+[:\s]", text_lower, 2, "appendix/annex"),
        ]
        for pattern, search_text, weight, desc in expert_patterns:
            if re.search(pattern, search_text):
                scores["expert_report"] += weight
                matched_patterns["expert_report"].append(desc)

        # =====================================================================
        # SCHEDULE OF LOSS / DAMAGES
        # =====================================================================
        schedule_patterns = [
            (r"schedule\s+of\s+(?:loss|damage|losses)", text_start, 8, "schedule of loss header"),
            (r"quantum\s+(?:of\s+)?(?:claim|damages)", text_start, 6, "quantum claim"),
            (r"(?:counter[-\s]?schedule|counter\s+schedule)", text_start, 6, "counter-schedule"),
            (r"(?:principal|interest|total)\s+claimed", text_lower, 4, "claimed amounts"),
            (r"loss\s+of\s+(?:profit|earnings|income)", text_lower, 3, "loss of profit"),
            (r"(?:special|general|consequential)\s+damages?", text_lower, 3, "damage types"),
            (r"schedule[-_]|sol[-_]", file_name, 4, "filename pattern"),
            (r"[\$£€]\s*[\d,]+(?:\.\d{2})?", text_lower, 2, "currency amounts"),
        ]
        for pattern, search_text, weight, desc in schedule_patterns:
            if re.search(pattern, search_text):
                scores["schedule_of_loss"] += weight
                matched_patterns["schedule_of_loss"].append(desc)

        # =====================================================================
        # COURT FILING (Orders, Judgments, Directions)
        # =====================================================================
        court_filing_patterns = [
            (r"(?:it\s+is|the\s+court)\s+(?:hereby\s+)?order(?:ed|s)", text_start, 7, "court orders"),
            (r"judgment\s+(?:of|in|dated)", text_start, 6, "judgment"),
            (r"upon\s+(?:reading|hearing|consideration)", text_start, 5, "upon reading"),
            (r"before\s+(?:the\s+)?(?:honourable|hon\.?)\s+(?:mr\s+)?justice", text_start, 5, "before justice"),
            (r"sealed\s+(?:by\s+)?(?:the\s+)?court", text_lower, 5, "court seal"),
            (r"(?:order|judgment|ruling|directions?)[-_]", file_name, 4, "filename pattern"),
            (r"dated\s+this\s+\d+", text_lower, 3, "dated this day"),
            (r"(?:peremptory|unless)\s+order", text_lower, 4, "unless order"),
            (r"costs?\s+(?:in\s+the\s+case|reserved|summarily)", text_lower, 3, "costs order"),
        ]
        for pattern, search_text, weight, desc in court_filing_patterns:
            if re.search(pattern, search_text):
                scores["court_filing"] += weight
                matched_patterns["court_filing"].append(desc)

        # =====================================================================
        # STATUTE / LEGISLATION
        # =====================================================================
        statute_patterns = [
            (r"an\s+act\s+to\s+(?:amend|provide|make)", text_start, 7, "An Act to"),
            (r"be\s+it\s+(?:enacted|resolved)", text_start, 7, "Be it enacted"),
            (r"(?:act|regulation)\s+\d{4}", text_start, 5, "Act year"),
            (r"section\s+\d+\s*\(\d+\)", text_lower, 5, "section numbering"),
            (r"(?:subsection|paragraph|subparagraph)", text_lower, 3, "legislative structure"),
            (r"(?:commencement|interpretation|extent)\s+(?:and\s+)?(?:application)?", text_start, 3, "statute sections"),
            (r"statute|legislation|_act[-_]", file_name, 4, "filename pattern"),
            (r"(?:enterprise\s+act|competition\s+act|fraud\s+act|companies\s+act)", text_lower, 5, "specific UK acts"),
        ]
        for pattern, search_text, weight, desc in statute_patterns:
            if re.search(pattern, search_text):
                scores["statute"] += weight
                matched_patterns["statute"].append(desc)

        # =====================================================================
        # CASE LAW / JUDGMENT TRANSCRIPT
        # =====================================================================
        case_law_patterns = [
            (r"\[\d{4}\]\s+(?:ewca|ewhc|uksc|ukhl)", text_start, 8, "neutral citation"),
            (r"(?:v\s+|versus\s+).{2,50}?\[\d{4}\]", text_start, 6, "case name with citation"),
            (r"approved\s+judgment", text_start, 6, "approved judgment"),
            (r"(?:lord|lady)\s+justice\s+[a-z]+", text_lower, 4, "Lord/Lady Justice"),
            (r"(?:para(?:graph)?s?\s+)?\[\d+\]", text_lower, 2, "paragraph citations"),
            (r"(?:claimant|appellant)\s+(?:v|versus)\s+", text_start, 4, "v case format"),
            (r"case[-_]?law|judgment[-_]", file_name, 4, "filename pattern"),
        ]
        for pattern, search_text, weight, desc in case_law_patterns:
            if re.search(pattern, search_text):
                scores["case_law"] += weight
                matched_patterns["case_law"].append(desc)

        # =====================================================================
        # CONTRACT / AGREEMENT
        # =====================================================================
        contract_patterns = [
            (r"this\s+(?:agreement|contract|deed)", text_start, 5, "this agreement"),
            (r"(?:parties?\s+)?(?:hereby\s+)?agree(?:s|d)?", text_start, 4, "parties agree"),
            (r"(?:whereas|recitals?|background)", text_start, 4, "whereas/recitals"),
            (r"(?:now\s+)?(?:therefore|it\s+is\s+agreed)", text_lower, 4, "now therefore"),
            (r"in\s+witness\s+whereof", text_lower, 5, "in witness whereof"),
            (r"executed\s+(?:as\s+)?a?\s*deed", text_lower, 5, "executed as deed"),
            (r"(?:obligations?|warranties|representations?|covenants?)", text_lower, 2, "contract terms"),
            (r"(?:termination|governing\s+law|jurisdiction)", text_lower, 2, "boilerplate"),
            (r"(?:contract|agreement|deed|nda|mou|sla)[-_]", file_name, 4, "filename pattern"),
            (r"(?:schedule|exhibit|annex)\s+\d+", text_lower, 2, "schedules"),
        ]
        for pattern, search_text, weight, desc in contract_patterns:
            if re.search(pattern, search_text):
                scores["contract"] += weight
                matched_patterns["contract"].append(desc)

        # =====================================================================
        # EMAIL
        # =====================================================================
        email_patterns = [
            (r"from:\s*[^\n]+", text_start, 5, "From: header"),
            (r"to:\s*[^\n]+", text_start, 4, "To: header"),
            (r"subject:\s*[^\n]+", text_start, 4, "Subject: header"),
            (r"(?:sent|date):\s*\d+", text_start, 3, "Sent/Date: header"),
            (r"cc:\s*[^\n]+", text_start, 3, "CC: header"),
            (r"\.eml$|\.msg$", file_path_str, 6, "email file extension"),
            (r"email[-_]|mail[-_]", file_name, 3, "filename pattern"),
            (r"-----\s*original\s+message\s*-----", text_lower, 5, "forwarded email"),
            (r"@[a-z0-9]+\.[a-z]{2,}", text_start, 2, "email addresses"),
        ]
        for pattern, search_text, weight, desc in email_patterns:
            if re.search(pattern, search_text):
                scores["email"] += weight
                matched_patterns["email"].append(desc)

        # =====================================================================
        # LETTER / CORRESPONDENCE
        # =====================================================================
        letter_patterns = [
            (r"(?:dear\s+(?:sir|madam|mr|mrs|ms|sirs))", text_start, 5, "Dear Sir/Madam"),
            (r"yours\s+(?:faithfully|sincerely)", text_lower, 5, "Yours faithfully"),
            (r"(?:without\s+prejudice|strictly\s+confidential)", text_start, 4, "without prejudice"),
            (r"we\s+(?:write|refer|act)\s+(?:to|for|on)", text_start, 3, "we write to"),
            (r"(?:solicitors?|counsel)\s+for", text_start, 3, "solicitors for"),
            (r"letter[-_]|correspondence[-_]", file_name, 3, "filename pattern"),
            (r"re:\s+[^\n]+", text_start, 2, "Re: subject line"),
            (r"(?:our|your)\s+(?:ref|reference)[:\s]", text_start, 3, "reference line"),
        ]
        for pattern, search_text, weight, desc in letter_patterns:
            if re.search(pattern, search_text):
                scores["letter"] += weight
                matched_patterns["letter"].append(desc)

        # =====================================================================
        # DISCLOSURE / EXHIBIT
        # =====================================================================
        disclosure_patterns = [
            (r"(?:exhibit|annex)\s+[a-z0-9]+", text_start, 4, "exhibit label"),
            (r"this\s+is\s+(?:the\s+)?exhibit", text_start, 5, "this is exhibit"),
            (r"disclosure[-_]|exhibit[-_]|bundle[-_]", file_name, 4, "filename pattern"),
            (r"paginated\s+bundle", text_lower, 4, "paginated bundle"),
            (r"(?:document|page)\s+\d+\s+of\s+\d+", text_lower, 3, "page numbering"),
        ]
        for pattern, search_text, weight, desc in disclosure_patterns:
            if re.search(pattern, search_text):
                scores["disclosure"] += weight
                matched_patterns["disclosure"].append(desc)

        # =====================================================================
        # DISCLOSURE LIST (Form N265 / Schedule)
        # =====================================================================
        disclosure_list_patterns = [
            (r"list\s+of\s+documents?", text_start, 7, "list of documents"),
            (r"form\s+n\s*265", text_start, 8, "Form N265"),
            (r"(?:standard|specific)\s+disclosure", text_start, 5, "standard/specific disclosure"),
            (r"(?:inspection|copies?)\s+(?:of\s+)?documents?", text_lower, 3, "inspection of documents"),
            (r"disclosure[-_]list|n265", file_name, 5, "filename pattern"),
            (r"part\s+[12]\s*[:\s]", text_start, 3, "Part 1/2 structure"),
        ]
        for pattern, search_text, weight, desc in disclosure_list_patterns:
            if re.search(pattern, search_text):
                scores["disclosure_list"] += weight
                matched_patterns["disclosure_list"].append(desc)

        # =====================================================================
        # COURT FORM (N-series forms)
        # =====================================================================
        court_form_patterns = [
            (r"form\s+n\s*\d+", text_start, 7, "Form N-series"),
            (r"(?:n1|n260|n244|n170|n180|n9)", text_start, 6, "specific form number"),
            (r"(?:claim\s+form|application\s+notice)", text_start, 5, "claim form"),
            (r"(?:allocation\s+questionnaire|directions\s+questionnaire)", text_start, 5, "questionnaire"),
            (r"form[-_]n\d+|n\d+[-_]", file_name, 5, "filename pattern"),
        ]
        for pattern, search_text, weight, desc in court_form_patterns:
            if re.search(pattern, search_text):
                scores["court_form"] += weight
                matched_patterns["court_form"].append(desc)

        # =====================================================================
        # CASE MANAGEMENT
        # =====================================================================
        case_mgmt_patterns = [
            (r"(?:case|costs?)\s+management\s+(?:conference|order|directions)", text_start, 7, "CMC/CMO"),
            (r"directions?\s+order", text_start, 5, "directions order"),
            (r"pre[-\s]?trial\s+(?:review|checklist|timetable)", text_start, 5, "PTR/checklist"),
            (r"(?:agreed\s+)?(?:directions|timetable)", text_start, 4, "agreed directions"),
            (r"cmc[-_]|ptc[-_]|directions[-_]", file_name, 4, "filename pattern"),
            (r"(?:disclosure|witness\s+statements?|expert)\s+by\s+\d+", text_lower, 3, "deadlines"),
        ]
        for pattern, search_text, weight, desc in case_mgmt_patterns:
            if re.search(pattern, search_text):
                scores["case_management"] += weight
                matched_patterns["case_management"].append(desc)

        # =====================================================================
        # CHRONOLOGY
        # =====================================================================
        chronology_patterns = [
            (r"chronology", text_start, 8, "chronology header"),
            (r"(?:agreed\s+)?(?:statement\s+of\s+)?(?:facts?\s+and\s+)?issues?", text_start, 4, "agreed facts"),
            (r"(?:date|period|time)[:\s]+(?:event|description|fact)", text_start, 5, "date/event columns"),
            (r"chronology[-_]|timeline[-_]", file_name, 5, "filename pattern"),
            (r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}[:\s]+", text_lower, 3, "date entries"),
        ]
        for pattern, search_text, weight, desc in chronology_patterns:
            if re.search(pattern, search_text):
                scores["chronology"] += weight
                matched_patterns["chronology"].append(desc)

        # =====================================================================
        # MEDICAL REPORT
        # =====================================================================
        medical_patterns = [
            (r"(?:medical|medico[-\s]?legal)\s+report", text_start, 7, "medical report header"),
            (r"(?:diagnosis|prognosis|treatment|history)", text_lower, 3, "medical terms"),
            (r"(?:gp|consultant|doctor|surgeon|psychiatrist|psychologist)", text_lower, 3, "medical professionals"),
            (r"(?:injury|injuries|condition|symptoms)", text_lower, 2, "injury terms"),
            (r"medical[-_]|medico[-_]", file_name, 4, "filename pattern"),
            (r"(?:examination|assessment)\s+(?:on|dated)", text_lower, 3, "examination date"),
        ]
        for pattern, search_text, weight, desc in medical_patterns:
            if re.search(pattern, search_text):
                scores["medical_report"] += weight
                matched_patterns["medical_report"].append(desc)

        # =====================================================================
        # TRIBUNAL DOCUMENT
        # =====================================================================
        tribunal_patterns = [
            (r"(?:employment|competition)\s+(?:tribunal|appeal\s+tribunal)", text_start, 7, "tribunal name"),
            (r"(?:cat|eat|et)\s+case\s+(?:no|number)", text_start, 6, "tribunal case number"),
            (r"competition\s+(?:and\s+markets?\s+)?authority|cma", text_lower, 5, "CMA reference"),
            (r"(?:tribunal|appeal)\s+(?:judgment|decision|ruling)", text_start, 5, "tribunal decision"),
            (r"tribunal[-_]|cat[-_]|eat[-_]", file_name, 4, "filename pattern"),
        ]
        for pattern, search_text, weight, desc in tribunal_patterns:
            if re.search(pattern, search_text):
                scores["tribunal_document"] += weight
                matched_patterns["tribunal_document"].append(desc)

        # =====================================================================
        # REGULATORY DOCUMENT
        # =====================================================================
        regulatory_patterns = [
            (r"(?:fca|pra|sra|ofcom|ofgem)\s+(?:decision|finding|notice)", text_start, 6, "regulator decision"),
            (r"(?:regulatory|enforcement)\s+(?:notice|action|decision)", text_start, 5, "regulatory notice"),
            (r"(?:investigation|enforcement)\s+(?:report|summary)", text_start, 4, "investigation report"),
            (r"(?:breach|contravention)\s+of\s+(?:rule|regulation)", text_lower, 4, "breach of rules"),
            (r"regulatory[-_]|fca[-_]|enforcement[-_]", file_name, 4, "filename pattern"),
        ]
        for pattern, search_text, weight, desc in regulatory_patterns:
            if re.search(pattern, search_text):
                scores["regulatory_document"] += weight
                matched_patterns["regulatory_document"].append(desc)

        # =====================================================================
        # DETERMINE WINNER
        # =====================================================================
        
        # Find the highest scoring type
        max_score = max(scores.values())
        
        # Calculate confidence (normalize against max possible)
        # Approximate max score is ~20 for a strong match
        max_possible = 25.0
        confidence = min(max_score / max_possible, 1.0)
        
        # If no strong signals, default to disclosure
        if max_score < 3:
            return ClassificationResult(
                document_type="disclosure",
                confidence=0.2,
                matched_patterns=["no strong patterns matched, defaulting to disclosure"]
            )
        
        # Get the winning type
        winner = "unknown"
        for doc_type, score in scores.items():
            if score == max_score:
                winner = doc_type
                break
        
        return ClassificationResult(
            document_type=winner,
            confidence=confidence,
            matched_patterns=matched_patterns[winner]
        )

    def _get_all_types(self) -> list[str]:
        """Get all supported document types."""
        return [
            "witness_statement",
            "court_filing",
            "pleading",
            "skeleton_argument",
            "expert_report",
            "schedule_of_loss",
            "statute",
            "case_law",
            "contract",
            "email",
            "letter",
            "disclosure",
            "disclosure_list",
            "court_form",
            "case_management",
            "chronology",
            "medical_report",
            "tribunal_document",
            "regulatory_document",
        ]

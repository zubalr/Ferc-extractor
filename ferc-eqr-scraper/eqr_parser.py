"""Custom parser for FERC EQR XML files (native format, not XBRL).

This parser handles FERC EQR XML filings which can contain:
- Multiple organizations (sellers, buyers, filers)
- Multiple contracts with products and transactions
- Contact information for organizations
- Large files (70MB+) with 400+ organizations and contracts

The parser uses efficient memory management for large files with lxml
for faster XML processing and proper namespace handling.
"""

try:
    from lxml import etree as ET
    USING_LXML = True
except ImportError:
    import xml.etree.ElementTree as ET
    USING_LXML = False
    
import pandas as pd
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import os


class EQRXMLParser:
    """Parser for FERC EQR XML files in native format.
    
    Handles large XML files efficiently with proper memory management
    and namespace support for the FERC EQR XML schema.
    Uses lxml for better performance when available.
    """
    
    def __init__(self):
        """Initialize the EQR XML parser."""
        self.logger = logging.getLogger("ferc_scraper.eqr_parser")
        self.namespace = {"eqr": "urn:www.ferc.gov:forms.eqr"}
        
        # Log which XML parser is being used
        if USING_LXML:
            self.logger.info("Using lxml for XML parsing (better performance)")
        else:
            self.logger.info("Using xml.etree.ElementTree for XML parsing (lxml not available)")
        
        # Track parsing statistics
        self.stats = {
            'files_processed': 0,
            'organizations_parsed': 0,
            'contacts_parsed': 0,
            'contracts_parsed': 0,
            'products_parsed': 0,
            'transactions_parsed': 0
        }
    
    def parse_xml_file(self, xml_path: str) -> Dict[str, pd.DataFrame]:
        """Parse a single EQR XML file into DataFrames.
        
        Efficiently handles large XML files with hundreds of organizations
        and contracts by using streaming-like parsing and memory management.
        Uses lxml when available for better performance.
        
        Args:
            xml_path: Path to the XML file
            
        Returns:
            Dictionary of DataFrames with parsed data:
            - 'organizations': Organization/company data
            - 'contacts': Contact information for organizations  
            - 'contracts': Contract data
            - 'contract_products': Products associated with contracts
            - 'transactions': Transaction data
            
        Raises:
            ET.ParseError: If XML parsing fails
            Exception: For other parsing errors
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Parsing EQR XML file: {os.path.basename(xml_path)}")
            
            # Parse the XML file with lxml optimizations if available
            if USING_LXML:
                # Use lxml's optimized parser
                parser = ET.XMLParser(recover=True, strip_cdata=False)
                tree = ET.parse(xml_path, parser)
            else:
                # Fall back to standard library
                tree = ET.parse(xml_path)
                
            root = tree.getroot()
            
            # Extract filing metadata
            filing_metadata = self._extract_filing_metadata(root)
            self.logger.debug(f"Filing metadata: {filing_metadata}")
            
            # Parse different sections
            dataframes = {}
            
            # Parse organizations (includes sellers, buyers, filers)
            organizations_df = self._parse_organizations(root, filing_metadata)
            if not organizations_df.empty:
                organizations_df = self._validate_and_clean_dataframe(organizations_df, 'organizations')
                if not organizations_df.empty:
                    dataframes['organizations'] = organizations_df
                    self.stats['organizations_parsed'] += len(organizations_df)
            
            # Parse contacts
            contacts_df = self._parse_contacts(root, filing_metadata)
            if not contacts_df.empty:
                contacts_df = self._validate_and_clean_dataframe(contacts_df, 'contacts')
                if not contacts_df.empty:
                    dataframes['contacts'] = contacts_df
                    self.stats['contacts_parsed'] += len(contacts_df)
            
            # Parse contracts
            contracts_df = self._parse_contracts(root, filing_metadata)
            if not contracts_df.empty:
                contracts_df = self._validate_and_clean_dataframe(contracts_df, 'contracts')
                if not contracts_df.empty:
                    dataframes['contracts'] = contracts_df
                    self.stats['contracts_parsed'] += len(contracts_df)
            
            # Parse contract products
            contract_products_df = self._parse_contract_products(root, filing_metadata)
            if not contract_products_df.empty:
                contract_products_df = self._validate_and_clean_dataframe(contract_products_df, 'contract_products')
                if not contract_products_df.empty:
                    dataframes['contract_products'] = contract_products_df
                    self.stats['products_parsed'] += len(contract_products_df)
            
            # Parse transactions
            transactions_df = self._parse_transactions(root, filing_metadata)
            if not transactions_df.empty:
                transactions_df = self._validate_and_clean_dataframe(transactions_df, 'transactions')
                if not transactions_df.empty:
                    dataframes['transactions'] = transactions_df
                    self.stats['transactions_parsed'] += len(transactions_df)
            
            # Log results
            processing_time = (datetime.now() - start_time).total_seconds()
            parser_info = " (lxml)" if USING_LXML else " (ElementTree)"
            self.logger.info(f"Successfully parsed {len(dataframes)} tables from XML file in {processing_time:.2f}s{parser_info}")
            
            for table_name, df in dataframes.items():
                self.logger.info(f"  {table_name}: {len(df)} rows")
            
            self.stats['files_processed'] += 1
            return dataframes
            
        except ET.ParseError as e:
            self.logger.error(f"XML parsing error in {xml_path}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error parsing {xml_path}: {e}")
            raise
    
    def parse_xml_file_auto(self, xml_path: str, file_size_threshold: int = None) -> Dict[str, pd.DataFrame]:
        """Ultimate memory-efficient XML parser with intelligent strategy selection.
        
        This is the ONE parser you need - it automatically chooses the best approach:
        - Small files (<20MB): Fast in-memory parsing
        - Large files (>20MB): Memory-efficient streaming with chunked processing
        - Emergency memory monitoring throughout
        
        Args:
            xml_path: Path to the XML file
            file_size_threshold: File size threshold in bytes (uses 20MB default)
            
        Returns:
            Dictionary of DataFrames with parsed data
        """
        start_time = datetime.now()
        
        try:
            file_size = os.path.getsize(xml_path)
            file_size_mb = file_size / (1024 * 1024)
            
            # Use 20MB threshold by default
            if file_size_threshold is None:
                file_size_threshold = 20 * 1024 * 1024  # 20MB
            
            self.logger.info(f"Processing {os.path.basename(xml_path)} ({file_size_mb:.1f}MB)")
            
            # Memory check before starting
            import psutil
            initial_memory = psutil.virtual_memory().percent
            if initial_memory > 75:
                self.logger.warning(f"High initial memory usage ({initial_memory:.1f}%), using conservative processing")
                file_size_threshold = 5 * 1024 * 1024  # Use 5MB threshold if memory is already high
            
            # Parse the file using the optimal strategy
            if file_size > file_size_threshold and USING_LXML:
                self.logger.info("Using memory-efficient streaming parser")
                dataframes = self._parse_xml_streaming_optimized(xml_path)
            else:
                self.logger.info("Using fast in-memory parser")
                dataframes = self._parse_xml_memory_optimized(xml_path)
            
            # Log results
            processing_time = (datetime.now() - start_time).total_seconds()
            final_memory = psutil.virtual_memory().percent
            memory_delta = final_memory - initial_memory
            
            self.logger.info(f"✅ Parsed {len(dataframes)} tables in {processing_time:.2f}s")
            self.logger.info(f"📊 Memory: {initial_memory:.1f}% → {final_memory:.1f}% (Δ{memory_delta:+.1f}%)")
            
            for table_name, df in dataframes.items():
                self.logger.info(f"   {table_name}: {len(df)} rows")
            
            return dataframes
            
        except Exception as e:
            self.logger.error(f"Failed to parse {xml_path}: {e}")
            raise
                
        except OSError:
            # If we can't get file size, use standard parser
            self.logger.warning(f"Could not determine file size for {xml_path}, using standard parser")
            return self._parse_xml_memory_optimized(xml_path)

    def _parse_xml_memory_optimized(self, xml_path: str) -> Dict[str, pd.DataFrame]:
        """Fast in-memory parser optimized for small-medium files (<20MB).
        
        Uses standard ElementTree parsing with memory-efficient DataFrame creation.
        Perfect for files with <50K transactions.
        """
        start_time = datetime.now()
        
        try:
            self.logger.debug(f"Fast parsing: {os.path.basename(xml_path)}")
            
            # Parse XML efficiently
            if USING_LXML:
                from lxml import etree as ET
                tree = ET.parse(xml_path)
            else:
                tree = ET.parse(xml_path)
                
            root = tree.getroot()
            
            # Extract filing metadata with quarter fix
            filing_metadata = self._extract_filing_metadata(root)
            
            # Parse all sections with optimized memory usage
            dataframes = {}
            
            # Parse organizations
            organizations_df = self._parse_organizations_fast(root, filing_metadata)
            if not organizations_df.empty:
                dataframes['organizations'] = organizations_df
                self.stats['organizations_parsed'] += len(organizations_df)
            
            # Parse contacts
            contacts_df = self._parse_contacts_fast(root, filing_metadata)
            if not contacts_df.empty:
                dataframes['contacts'] = contacts_df
                self.stats['contacts_parsed'] += len(contacts_df)
            
            # Parse contracts
            contracts_df = self._parse_contracts_fast(root, filing_metadata)
            if not contracts_df.empty:
                dataframes['contracts'] = contracts_df
                self.stats['contracts_parsed'] += len(contracts_df)
            
            # Parse contract products
            contract_products_df = self._parse_contract_products_fast(root, filing_metadata)
            if not contract_products_df.empty:
                dataframes['contract_products'] = contract_products_df
                self.stats['products_parsed'] += len(contract_products_df)
            
            # Parse transactions with chunking for memory safety
            transactions_df = self._parse_transactions_chunked(root, filing_metadata)
            if not transactions_df.empty:
                dataframes['transactions'] = transactions_df
                self.stats['transactions_parsed'] += len(transactions_df)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.debug(f"Fast parsing completed in {processing_time:.2f}s")
            
            return dataframes
            
        except Exception as e:
            self.logger.error(f"Fast parsing error in {xml_path}: {e}")
            raise

    def _parse_xml_streaming_optimized(self, xml_path: str) -> Dict[str, pd.DataFrame]:
        """Memory-efficient streaming parser for large files (>20MB).
        
        Uses lxml iterparse to process XML without loading entire file into memory.
        Perfect for files with 100K+ transactions.
        """
        if not USING_LXML:
            self.logger.warning("Streaming requires lxml, falling back to memory parser")
            return self._parse_xml_memory_optimized(xml_path)
        
        start_time = datetime.now()
        
        try:
            from lxml import etree as ET
            self.logger.debug(f"Streaming parse: {os.path.basename(xml_path)}")
            
            # Initialize collectors
            organizations_data = []
            contacts_data = []
            contracts_data = []
            contract_products_data = []
            transaction_chunks = []
            
            filing_metadata = {}
            
            # Memory-efficient streaming parser
            context = ET.iterparse(xml_path, events=('start', 'end'), huge_tree=True)
            context = iter(context)
            event, root = next(context)
            
            # Extract filing metadata from root
            filing_metadata = self._extract_filing_metadata(root)
            
            current_contract = None
            transaction_buffer = []
            TRANSACTION_CHUNK_SIZE = 10000  # Process 10K transactions at a time
            
            for event, elem in context:
                if event == 'end':
                    # Process different element types
                    tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
                    
                    if tag_name == 'Organization':
                        org_data = self._extract_organization_data(elem, filing_metadata)
                        if org_data:
                            organizations_data.append(org_data)
                    
                    elif tag_name == 'Contact':
                        contact_data = self._extract_contact_data(elem, filing_metadata)
                        if contact_data:
                            contacts_data.append(contact_data)
                    
                    elif tag_name == 'Contract':
                        current_contract = elem.get('Uid')
                        contract_data = self._extract_contract_data(elem, filing_metadata)
                        if contract_data:
                            contracts_data.append(contract_data)
                    
                    elif tag_name == 'ContractProduct':
                        product_data = self._extract_contract_product_data(elem, filing_metadata, current_contract)
                        if product_data:
                            contract_products_data.append(product_data)
                    
                    elif tag_name == 'Transaction':
                        transaction_data = self._extract_transaction_data(elem, filing_metadata, current_contract)
                        if transaction_data:
                            transaction_buffer.append(transaction_data)
                            
                            # Process transactions in chunks to limit memory
                            if len(transaction_buffer) >= TRANSACTION_CHUNK_SIZE:
                                chunk_df = pd.DataFrame(transaction_buffer)
                                chunk_df = self._process_transaction_chunk(chunk_df)
                                transaction_chunks.append(chunk_df)
                                transaction_buffer = []
                                
                                # Force cleanup
                                import gc
                                gc.collect()
                    
                    # Clear processed element to free memory
                    elem.clear()
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]
            
            # Process remaining transactions
            if transaction_buffer:
                chunk_df = pd.DataFrame(transaction_buffer)
                chunk_df = self._process_transaction_chunk(chunk_df)
                transaction_chunks.append(chunk_df)
            
            # Create final DataFrames
            dataframes = {}
            
            if organizations_data:
                dataframes['organizations'] = pd.DataFrame(organizations_data)
                self.stats['organizations_parsed'] += len(organizations_data)
            
            if contacts_data:
                dataframes['contacts'] = pd.DataFrame(contacts_data)
                self.stats['contacts_parsed'] += len(contacts_data)
            
            if contracts_data:
                dataframes['contracts'] = pd.DataFrame(contracts_data)
                self.stats['contracts_parsed'] += len(contracts_data)
            
            if contract_products_data:
                dataframes['contract_products'] = pd.DataFrame(contract_products_data)
                self.stats['products_parsed'] += len(contract_products_data)
            
            if transaction_chunks:
                dataframes['transactions'] = pd.concat(transaction_chunks, ignore_index=True)
                self.stats['transactions_parsed'] += len(dataframes['transactions'])
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.debug(f"Streaming parse completed in {processing_time:.2f}s")
            
            return dataframes
            
        except Exception as e:
            self.logger.error(f"Streaming parse error in {xml_path}: {e}")
            # Fallback to memory parser
            return self._parse_xml_memory_optimized(xml_path)

    def parse_xml_file_streaming(self, xml_path: str) -> Dict[str, pd.DataFrame]:
        """Parse XML file using streaming approach for maximum memory efficiency.
        
        This method uses lxml's iterparse for streaming XML processing,
        which is ideal for very large XML files (100MB+).
        
        Args:
            xml_path: Path to the XML file
            
        Returns:
            Dictionary of DataFrames with parsed data
        """
        if not USING_LXML:
            self.logger.warning("Streaming parser requires lxml, falling back to standard parsing")
            return self.parse_xml_file(xml_path)
        
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Streaming parse of EQR XML file: {os.path.basename(xml_path)}")
            
            # Initialize data collectors
            organizations = []
            contacts = []
            contracts = []
            
            filing_metadata = {}
            current_organization = None
            
            # Streaming XML parser
            context = ET.iterparse(xml_path, events=('start', 'end'))
            context = iter(context)
            event, root = next(context)
            
            # Extract filing metadata from root
            if not filing_metadata:
                filing_metadata = {
                    'filing_uid': root.get('Uid'),
                    'period_type': root.get('PeriodType'),
                    'year': int(root.get('Year')) if root.get('Year') else None,
                    'quarter': self._convert_quarter_to_int(root.get('Quarter')),
                    'filing_type': root.get('Type'),
                    'parsed_at': datetime.now().isoformat()
                }
            
            for event, elem in context:
                if event == 'end':
                    # Process organizations
                    if elem.tag.endswith('}Organization') or elem.tag == 'Organization':
                        org_data = {
                            'organization_uid': elem.get('Uid'),
                            'cid': elem.get('Cid'),
                            'company_name': elem.get('Name'),
                            'is_filer': elem.get('IsFiler') == 'true',
                            'is_buyer': elem.get('IsBuyer') == 'true',
                            'is_seller': elem.get('IsSeller') == 'true',
                            'transactions_reported_to_index_publisher': elem.get('TransactionsReportedToIndexPublisher') == 'true',
                            **filing_metadata
                        }
                        organizations.append(org_data)
                        current_organization = org_data['organization_uid']
                    
                    # Process contacts within organization
                    elif (elem.tag.endswith('}Contact') or elem.tag == 'Contact') and current_organization:
                        contact_data = {
                            'contact_uid': elem.get('Uid'),
                            'organization_uid': current_organization,
                            'first_name': elem.get('FirstName'),
                            'last_name': elem.get('LastName'),
                            'display_name': elem.get('DisplayName'),
                            'title': elem.get('Title'),
                            'phone': elem.get('Phone'),
                            'email': elem.get('Email'),
                            'is_filer_contact': elem.get('IsFilerContact') == 'true',
                            'is_buyer_contact': elem.get('IsBuyerContact') == 'true',
                            'is_seller_contact': elem.get('IsSellerContact') == 'true',
                            **filing_metadata
                        }
                        
                        # Look for address within contact
                        address_elem = elem.find('.//{*}Address') or elem.find('.//Address')
                        if address_elem is not None:
                            contact_data.update({
                                'street1': address_elem.get('Street1'),
                                'street2': address_elem.get('Street2'),
                                'street3': address_elem.get('Street3'),
                                'city': address_elem.get('City'),
                                'state': address_elem.get('State'),
                                'zip': address_elem.get('Zip'),
                                'country': address_elem.get('Country')
                            })
                        
                        contacts.append(contact_data)
                    
                    # Process contracts
                    elif elem.tag.endswith('}Contract') or elem.tag == 'Contract':
                        contract_data = {
                            'contract_uid': elem.get('Uid'),
                            'seller_uid': elem.get('SellerUid'),
                            'buyer_uid': elem.get('BuyerUid'),
                            'ferc_tariff_reference': elem.get('FercTariffReference'),
                            'contract_service_agreement': elem.get('ContractServiceAgreement'),
                            'is_affiliate': elem.get('IsAffiliate') == 'true',
                            'execution_date': elem.get('ExecutionDate'),
                            'commencement_date': elem.get('CommencementDate'),
                            'termination_date': elem.get('TerminationDate'),
                            'extension_provision_description': elem.get('ExtensionProvisionDescription'),
                            'filing_type_contract': elem.get('FilingType'),
                            **filing_metadata
                        }
                        contracts.append(contract_data)
                    
                    # Clear processed elements to free memory
                    elem.clear()
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]
            
            # Convert to DataFrames
            dataframes = {}
            
            if organizations:
                df = pd.DataFrame(organizations)
                df = self._validate_and_clean_dataframe(df, 'organizations')
                if not df.empty:
                    dataframes['organizations'] = df
                    self.stats['organizations_parsed'] += len(df)
            
            if contacts:
                df = pd.DataFrame(contacts)
                df = self._validate_and_clean_dataframe(df, 'contacts')
                if not df.empty:
                    dataframes['contacts'] = df
                    self.stats['contacts_parsed'] += len(df)
            
            if contracts:
                df = pd.DataFrame(contracts)
                df = self._validate_and_clean_dataframe(df, 'contracts')
                if not df.empty:
                    # Convert date columns
                    date_columns = ['execution_date', 'commencement_date', 'termination_date']
                    for col in date_columns:
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                    dataframes['contracts'] = df
                    self.stats['contracts_parsed'] += len(df)
            
            # Log results
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Streaming parse completed in {processing_time:.2f}s: {len(dataframes)} tables")
            
            for table_name, df in dataframes.items():
                self.logger.info(f"  {table_name}: {len(df)} rows")
            
            self.stats['files_processed'] += 1
            return dataframes
            
        except Exception as e:
            self.logger.error(f"Error in streaming parse of {xml_path}: {e}")
            # Fall back to regular parsing
            self.logger.info("Falling back to regular XML parsing")
            return self.parse_xml_file(xml_path)

    def _convert_quarter_to_int(self, quarter_str: str) -> Optional[int]:
        """Convert quarter text to integer.
        
        Args:
            quarter_str: Quarter as string ("First", "Second", etc. or "1", "2", etc.)
            
        Returns:
            Quarter as integer (1-4) or None if invalid
        """
        if not quarter_str:
            return None
        quarter_map = {
            'First': 1, 'first': 1, '1': 1,
            'Second': 2, 'second': 2, '2': 2, 
            'Third': 3, 'third': 3, '3': 3,
            'Fourth': 4, 'fourth': 4, '4': 4
        }
        return quarter_map.get(quarter_str, int(quarter_str) if quarter_str.isdigit() else None)

    def _extract_filing_metadata(self, root: ET.Element) -> Dict[str, Any]:
        """Extract filing-level metadata.
        
        Args:
            root: Root XML element
            
        Returns:
            Dictionary with filing metadata
        """
        metadata = {
            'filing_uid': root.get('Uid'),
            'period_type': root.get('PeriodType'),
            'year': int(root.get('Year')) if root.get('Year') else None,
            'quarter': self._convert_quarter_to_int(root.get('Quarter')),
            'filing_type': root.get('Type'),
            'parsed_at': datetime.now().isoformat()
        }
        
        # Validate critical metadata
        if not metadata['year'] or not metadata['quarter']:
            self.logger.warning(f"Missing year or quarter in filing metadata: {metadata}")
        
        self.logger.debug(f"Filing metadata: {metadata}")
        return metadata
        """Parse XML file using streaming approach for maximum memory efficiency.
        
        This method uses lxml's iterparse for streaming XML processing,
        which is ideal for very large XML files (100MB+).
        
        Args:
            xml_path: Path to the XML file
            
        Returns:
            Dictionary of DataFrames with parsed data
        """
        if not USING_LXML:
            self.logger.warning("Streaming parser requires lxml, falling back to standard parsing")
            return self.parse_xml_file(xml_path)
        
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Streaming parse of EQR XML file: {os.path.basename(xml_path)}")
            
            # Initialize data collectors
            organizations = []
            contacts = []
            contracts = []
            contract_products = []
            transactions = []
            
            filing_metadata = {}
            current_organization = None
            current_contract = None
            
            # Streaming XML parser
            context = ET.iterparse(xml_path, events=('start', 'end'))
            context = iter(context)
            event, root = next(context)
            
            # Extract filing metadata from root
            if not filing_metadata:
                filing_metadata = {
                    'filing_uid': root.get('Uid'),
                    'period_type': root.get('PeriodType'),
                    'year': int(root.get('Year')) if root.get('Year') else None,
                    'quarter': self._convert_quarter_to_int(root.get('Quarter')),
                    'filing_type': root.get('Type'),
                    'parsed_at': datetime.now().isoformat()
                }
            
            for event, elem in context:
                if event == 'end':
                    # Process organizations
                    if elem.tag.endswith('}Organization') or elem.tag == 'Organization':
                        org_data = {
                            'organization_uid': elem.get('Uid'),
                            'cid': elem.get('Cid'),
                            'company_name': elem.get('Name'),
                            'is_filer': elem.get('IsFiler') == 'true',
                            'is_buyer': elem.get('IsBuyer') == 'true',
                            'is_seller': elem.get('IsSeller') == 'true',
                            'transactions_reported_to_index_publisher': elem.get('TransactionsReportedToIndexPublisher') == 'true',
                            **filing_metadata
                        }
                        organizations.append(org_data)
                        current_organization = org_data['organization_uid']
                    
                    # Process contacts within organization
                    elif (elem.tag.endswith('}Contact') or elem.tag == 'Contact') and current_organization:
                        contact_data = {
                            'contact_uid': elem.get('Uid'),
                            'organization_uid': current_organization,
                            'first_name': elem.get('FirstName'),
                            'last_name': elem.get('LastName'),
                            'display_name': elem.get('DisplayName'),
                            'title': elem.get('Title'),
                            'phone': elem.get('Phone'),
                            'email': elem.get('Email'),
                            'is_filer_contact': elem.get('IsFilerContact') == 'true',
                            'is_buyer_contact': elem.get('IsBuyerContact') == 'true',
                            'is_seller_contact': elem.get('IsSellerContact') == 'true',
                            **filing_metadata
                        }
                        
                        # Look for address within contact
                        address_elem = elem.find('.//{*}Address') or elem.find('.//Address')
                        if address_elem is not None:
                            contact_data.update({
                                'street1': address_elem.get('Street1'),
                                'street2': address_elem.get('Street2'),
                                'street3': address_elem.get('Street3'),
                                'city': address_elem.get('City'),
                                'state': address_elem.get('State'),
                                'zip': address_elem.get('Zip'),
                                'country': address_elem.get('Country')
                            })
                        
                        contacts.append(contact_data)
                    
                    # Process contracts
                    elif elem.tag.endswith('}Contract') or elem.tag == 'Contract':
                        contract_data = {
                            'contract_uid': elem.get('Uid'),
                            'seller_uid': elem.get('SellerUid'),
                            'buyer_uid': elem.get('BuyerUid'),
                            'ferc_tariff_reference': elem.get('FercTariffReference'),
                            'contract_service_agreement': elem.get('ContractServiceAgreement'),
                            'is_affiliate': elem.get('IsAffiliate') == 'true',
                            'execution_date': elem.get('ExecutionDate'),
                            'commencement_date': elem.get('CommencementDate'),
                            'termination_date': elem.get('TerminationDate'),
                            'extension_provision_description': elem.get('ExtensionProvisionDescription'),
                            'filing_type_contract': elem.get('FilingType'),
                            **filing_metadata
                        }
                        contracts.append(contract_data)
                        current_contract = contract_data['contract_uid']
                    
                    # Clear processed elements to free memory
                    elem.clear()
                    while elem.getprevious() is not None:
                        del elem.getparent()[0]
            
            # Convert to DataFrames
            dataframes = {}
            
            if organizations:
                df = pd.DataFrame(organizations)
                df = self._validate_and_clean_dataframe(df, 'organizations')
                if not df.empty:
                    dataframes['organizations'] = df
                    self.stats['organizations_parsed'] += len(df)
            
            if contacts:
                df = pd.DataFrame(contacts)
                df = self._validate_and_clean_dataframe(df, 'contacts')
                if not df.empty:
                    dataframes['contacts'] = df
                    self.stats['contacts_parsed'] += len(df)
            
            if contracts:
                df = pd.DataFrame(contracts)
                df = self._validate_and_clean_dataframe(df, 'contracts')
                if not df.empty:
                    # Convert date columns
                    date_columns = ['execution_date', 'commencement_date', 'termination_date']
                    for col in date_columns:
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                    dataframes['contracts'] = df
                    self.stats['contracts_parsed'] += len(df)
            
            # Log results
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Streaming parse completed in {processing_time:.2f}s: {len(dataframes)} tables")
            
            for table_name, df in dataframes.items():
                self.logger.info(f"  {table_name}: {len(df)} rows")
            
            self.stats['files_processed'] += 1
            return dataframes
            
        except Exception as e:
            self.logger.error(f"Error in streaming parse of {xml_path}: {e}")
            # Fall back to regular parsing
            self.logger.info("Falling back to regular XML parsing")
            return self.parse_xml_file(xml_path)

    def _validate_and_clean_dataframe(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Validate and clean DataFrame data.
        
        Args:
            df: DataFrame to validate and clean
            table_name: Name of the table for logging
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        initial_rows = len(df)
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Clean string columns - remove excessive whitespace and empty strings
        for col in df.select_dtypes(include=['object']).columns:
            if col in df.columns:
                # Strip whitespace and convert empty strings to None
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace('', None)
                df[col] = df[col].replace('nan', None)
        
        # Ensure required fields are present for each table type
        required_fields = {
            'organizations': ['organization_uid'],
            'contacts': ['contact_uid', 'organization_uid'],
            'contracts': ['contract_uid'],
            'contract_products': ['product_uid'],
            'transactions': ['transaction_uid']
        }
        
        if table_name in required_fields:
            for field in required_fields[table_name]:
                if field in df.columns:
                    # Remove rows where required fields are null
                    before_count = len(df)
                    df = df.dropna(subset=[field])
                    after_count = len(df)
                    if before_count != after_count:
                        self.logger.info(f"Removed {before_count - after_count} rows with null {field} from {table_name}")
        
        cleaned_rows = len(df)
        if cleaned_rows != initial_rows:
            self.logger.info(f"Cleaned {table_name}: {initial_rows} → {cleaned_rows} rows")
        
        return df
    
    def _parse_companies(self, root: ET.Element, filing_metadata: Dict[str, Any]) -> pd.DataFrame:
        """Parse companies (organizations) from the XML.
        
        This method is kept for backward compatibility but delegates to
        the new _parse_organizations method.
        
        Args:
            root: Root XML element
            filing_metadata: Filing metadata dictionary
            
        Returns:
            DataFrame with company/organization data
        """
        return self._parse_organizations(root, filing_metadata)
    
    def _parse_organizations(self, root: ET.Element, filing_metadata: Dict[str, Any]) -> pd.DataFrame:
        """Parse organizations (companies/entities) from the XML.
        
        Organizations can be filers, buyers, sellers, or combinations thereof.
        Each organization has a unique UID and may have a CID (Company ID).
        
        Args:
            root: Root XML element
            filing_metadata: Filing metadata dictionary
            
        Returns:
            DataFrame with organization data
        """
        organizations = []
        
        # Parse organizations using proper namespace
        org_elements = root.findall('eqr:Organizations/eqr:Organization', self.namespace)
        
        for org in org_elements:
            org_data = {
                'organization_uid': org.get('Uid'),
                'cid': org.get('Cid'),  # Company ID (may be None)
                'company_name': org.get('Name'),
                'is_filer': org.get('IsFiler') == 'true',
                'is_buyer': org.get('IsBuyer') == 'true',
                'is_seller': org.get('IsSeller') == 'true',
                'transactions_reported_to_index_publisher': org.get('TransactionsReportedToIndexPublisher') == 'true',
                'filing_uid': filing_metadata['filing_uid'],
                'year': filing_metadata['year'],
                'quarter': filing_metadata['quarter'],
                'period_type': filing_metadata['period_type'],
                'filing_type': filing_metadata['filing_type']
            }
            organizations.append(org_data)
        
        df = pd.DataFrame(organizations)
        if not df.empty:
            self.logger.debug(f"Parsed {len(df)} organizations")
        
        return df
    
    def _parse_contacts(self, root: ET.Element, filing_metadata: Dict[str, Any]) -> pd.DataFrame:
        """Parse contacts from organizations in the XML.
        
        Each organization can have multiple contacts with different roles
        (filer contact, buyer contact, seller contact).
        
        Args:
            root: Root XML element
            filing_metadata: Filing metadata dictionary
            
        Returns:
            DataFrame with contact data including addresses
        """
        contacts = []
        
        # Parse contacts from organizations using proper namespace
        organizations = root.findall('eqr:Organizations/eqr:Organization', self.namespace)
        for org in organizations:
            contact_elements = org.findall('eqr:Contacts/eqr:Contact', self.namespace)
            for contact in contact_elements:
                contact_data = {
                    'contact_uid': contact.get('Uid'),
                    'organization_uid': org.get('Uid'),
                    'first_name': contact.get('FirstName'),
                    'last_name': contact.get('LastName'),
                    'display_name': contact.get('DisplayName'),
                    'title': contact.get('Title'),
                    'phone': contact.get('Phone'),
                    'email': contact.get('Email'),
                    'is_filer_contact': contact.get('IsFilerContact') == 'true',
                    'is_buyer_contact': contact.get('IsBuyerContact') == 'true',
                    'is_seller_contact': contact.get('IsSellerContact') == 'true',
                    'filing_uid': filing_metadata['filing_uid'],
                    'year': filing_metadata['year'],
                    'quarter': filing_metadata['quarter'],
                    'period_type': filing_metadata['period_type'],
                    'filing_type': filing_metadata['filing_type']
                }
                
                # Parse address using namespace
                address = contact.find('eqr:Address', self.namespace)
                if address is not None:
                    contact_data.update({
                        'street1': address.get('Street1'),
                        'street2': address.get('Street2'),
                        'street3': address.get('Street3'),
                        'city': address.get('City'),
                        'state': address.get('State'),
                        'zip': address.get('Zip'),
                        'country': address.get('Country')
                    })
                
                contacts.append(contact_data)
        
        df = pd.DataFrame(contacts)
        if not df.empty:
            self.logger.debug(f"Parsed {len(df)} contacts")
        
        return df
    
    def _parse_contracts(self, root: ET.Element, filing_metadata: Dict[str, Any]) -> pd.DataFrame:
        """Parse contracts from the XML.
        
        Contracts link sellers and buyers and contain products and transactions.
        Each contract has execution dates, termination dates, and tariff references.
        
        Args:
            root: Root XML element
            filing_metadata: Filing metadata dictionary
            
        Returns:
            DataFrame with contract data
        """
        contracts = []
        
        # Parse contracts using proper namespace
        contract_elements = root.findall('eqr:Contracts/eqr:Contract', self.namespace)
        for contract in contract_elements:
            contract_data = {
                'contract_uid': contract.get('Uid'),
                'seller_uid': contract.get('SellerUid'),
                'buyer_uid': contract.get('BuyerUid'),
                'ferc_tariff_reference': contract.get('FercTariffReference'),
                'contract_service_agreement': contract.get('ContractServiceAgreement'),
                'is_affiliate': contract.get('IsAffiliate') == 'true',
                'execution_date': contract.get('ExecutionDate'),
                'commencement_date': contract.get('CommencementDate'),
                'termination_date': contract.get('TerminationDate'),
                'extension_provision_description': contract.get('ExtensionProvisionDescription'),
                'filing_type_contract': contract.get('FilingType'),  # Contract-level filing type
                'filing_uid': filing_metadata['filing_uid'],
                'year': filing_metadata['year'],
                'quarter': filing_metadata['quarter'],
                'period_type': filing_metadata['period_type'],
                'filing_type': filing_metadata['filing_type']  # Filing-level filing type
            }
            contracts.append(contract_data)
        
        df = pd.DataFrame(contracts)
        if not df.empty:
            self.logger.debug(f"Parsed {len(df)} contracts")
            
            # Convert date columns
            date_columns = ['execution_date', 'commencement_date', 'termination_date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def _parse_contract_products(self, root: ET.Element, filing_metadata: Dict[str, Any]) -> pd.DataFrame:
        """Parse contract products from the XML.
        
        Contract products define what is being traded under each contract,
        including capacity, energy, or other products with rates and terms.
        
        Args:
            root: Root XML element
            filing_metadata: Filing metadata dictionary
            
        Returns:
            DataFrame with contract product data
        """
        products = []
        
        # More efficient approach: iterate through contracts and their products
        contract_elements = root.findall('eqr:Contracts/eqr:Contract', self.namespace)
        
        for contract in contract_elements:
            contract_uid = contract.get('Uid')
            
            # Find contract products within this contract
            product_elements = contract.findall('.//eqr:ContractProduct', self.namespace)
            
            for product in product_elements:
                # Extract rate information using namespace
                rate_element = product.find('eqr:Rate', self.namespace)
                rate_description = rate_element.get('Description') if rate_element is not None else None
                rate_units = rate_element.get('Units') if rate_element is not None else None
                
                product_data = {
                    'product_uid': product.get('Uid'),
                    'contract_uid': contract_uid,
                    'product_type': product.get('ProductType'),
                    'product_name': product.get('ProductName'),
                    'product_class': product.get('Class'),
                    'term': product.get('Term'),
                    'increment': product.get('Increment'),
                    'increment_peaking': product.get('IncrementPeaking'),
                    'quantity': product.get('Quantity'),
                    'units': product.get('Units'),
                    'podsl': product.get('Podsl'),  # Point of delivery/receipt
                    'begin_date': product.get('BeginDate'),
                    'end_date': product.get('EndDate'),
                    'rate_description': rate_description,
                    'rate_units': rate_units,
                    'filing_type_product': product.get('FilingType'),  # Product-level filing type
                    'filing_uid': filing_metadata['filing_uid'],
                    'year': filing_metadata['year'],
                    'quarter': filing_metadata['quarter'],
                    'period_type': filing_metadata['period_type'],
                    'filing_type': filing_metadata['filing_type']  # Filing-level filing type
                }
                products.append(product_data)
        
        df = pd.DataFrame(products)
        if not df.empty:
            self.logger.debug(f"Parsed {len(df)} contract products")
            
            # Convert numeric columns
            numeric_columns = ['quantity']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert date columns
            date_columns = ['begin_date', 'end_date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    def _parse_transactions(self, root: ET.Element, filing_metadata: Dict[str, Any]) -> pd.DataFrame:
        """Parse transactions from contracts in the XML with memory-efficient chunking.
        
        Transactions are nested within contracts and contain pricing, quantity,
        and timing information. This method efficiently handles large files
        with thousands of transactions by processing them in chunks.
        
        Args:
            root: Root XML element
            filing_metadata: Filing metadata dictionary
            
        Returns:
            DataFrame with transaction data
        """
        from settings import Config
        settings = Config()
        max_rows = settings.MAX_TRANSACTION_ROWS_MEMORY
        
        transactions = []
        transaction_dfs = []  # Store chunked DataFrames
        
        # More efficient approach: iterate through contracts and their transactions
        contract_elements = root.findall('eqr:Contracts/eqr:Contract', self.namespace)
        
        total_transactions = 0
        for contract in contract_elements:
            contract_uid = contract.get('Uid')
            
            # Find transactions within this contract
            transaction_elements = contract.findall('.//eqr:Transaction', self.namespace)
            total_transactions += len(transaction_elements)
            
            for transaction in transaction_elements:
                transaction_data = {
                    'transaction_uid': transaction.get('Uid'),
                    'contract_uid': contract_uid,
                    'transaction_group_ref': transaction.get('TransactionGroupRef'),
                    'begin_date': transaction.get('BeginDate'),
                    'end_date': transaction.get('EndDate'),
                    'time_zone': transaction.get('TimeZone'),
                    'trade_date': transaction.get('TradeDate'),
                    'podba': transaction.get('Podba'),
                    'podsl': transaction.get('Podsl'),
                    'transaction_class': transaction.get('Class'),
                    'term': transaction.get('Term'),
                    'increment': transaction.get('Increment'),
                    'increment_peaking': transaction.get('IncrementPeaking'),
                    'product_name': transaction.get('ProductName'),
                    'quantity': transaction.get('Quantity'),
                    'standardized_quantity': transaction.get('StandardizedQuantity'),
                    'price': transaction.get('Price'),
                    'standardized_price': transaction.get('StandardizedPrice'),
                    'rate_units': transaction.get('RateUnits'),
                    'rate_type': transaction.get('RateType'),
                    'total_transmission_charge': transaction.get('TotalTransmissionCharge'),
                    'transaction_charge': transaction.get('TransactionCharge'),
                    'filing_type_transaction': transaction.get('FilingType'),  # Transaction-level filing type
                    'filing_uid': filing_metadata['filing_uid'],
                    'year': filing_metadata['year'],
                    'quarter': filing_metadata['quarter'],
                    'period_type': filing_metadata['period_type'],
                    'filing_type': filing_metadata['filing_type']  # Filing-level filing type
                }
                transactions.append(transaction_data)
                
                # Process in chunks to limit memory usage
                if len(transactions) >= max_rows:
                    self.logger.info(f"Processing transaction chunk: {len(transactions)} rows")
                    df_chunk = pd.DataFrame(transactions)
                    if not df_chunk.empty:
                        df_chunk = self._process_transaction_chunk(df_chunk)
                        transaction_dfs.append(df_chunk)
                    transactions = []  # Clear the list
                    
                    # Force garbage collection to free memory
                    import gc
                    gc.collect()
        
        # Process remaining transactions
        if transactions:
            self.logger.info(f"Processing final transaction chunk: {len(transactions)} rows")
            df_chunk = pd.DataFrame(transactions)
            if not df_chunk.empty:
                df_chunk = self._process_transaction_chunk(df_chunk)
                transaction_dfs.append(df_chunk)
        
        # Combine all chunks efficiently
        if transaction_dfs:
            self.logger.info(f"Combining {len(transaction_dfs)} transaction chunks (total: {total_transactions} transactions)")
            df = pd.concat(transaction_dfs, ignore_index=True)
            self.logger.debug(f"Final combined DataFrame has {len(df)} transactions")
        else:
            df = pd.DataFrame()
        
        return df
        
    def _process_transaction_chunk(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a chunk of transactions (convert numeric columns, etc.)."""
        if df.empty:
            return df
            
        # Convert numeric columns
        numeric_columns = [
            'quantity', 'standardized_quantity', 'price', 'standardized_price',
            'total_transmission_charge', 'transaction_charge'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def parse_multiple_files(self, xml_files: List[str]) -> Dict[str, List[pd.DataFrame]]:
        """Parse multiple XML files and combine results.
        
        Automatically chooses the best parsing strategy for each file.
        
        Args:
            xml_files: List of XML file paths
            
        Returns:
            Dictionary mapping table names to lists of DataFrames
        """
        all_dataframes = {}
        successful_files = 0
        failed_files = 0
        
        self.logger.info(f"Parsing {len(xml_files)} EQR XML files with auto strategy selection")
        
        for xml_file in xml_files:
            try:
                # Use auto parser selection based on file size
                dataframes = self.parse_xml_file_auto(xml_file)
                
                # Merge dataframes by table name
                for table_name, df in dataframes.items():
                    if table_name not in all_dataframes:
                        all_dataframes[table_name] = []
                    all_dataframes[table_name].append(df)
                
                successful_files += 1
                
            except Exception as e:
                self.logger.error(f"Failed to parse {xml_file}: {e}")
                failed_files += 1
                continue
        
        self.logger.info(
            f"Parsing complete: {successful_files} successful, {failed_files} failed. "
            f"Found {len(all_dataframes)} table types."
        )
        
        return all_dataframes
    
    def get_parsing_stats(self) -> Dict[str, int]:
        """Get statistics about parsing performance.
        
        Returns:
            Dictionary with parsing statistics
        """
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset parsing statistics."""
        for key in self.stats:
            self.stats[key] = 0
    
    # === FAST PARSING METHODS (for small files) ===
    
    def _parse_organizations_fast(self, root: ET.Element, filing_metadata: Dict[str, Any]) -> pd.DataFrame:
        """Fast organization parsing for small files."""
        return self._parse_organizations(root, filing_metadata)  # Reuse existing method
    
    def _parse_contacts_fast(self, root: ET.Element, filing_metadata: Dict[str, Any]) -> pd.DataFrame:
        """Fast contact parsing for small files."""
        return self._parse_contacts(root, filing_metadata)  # Reuse existing method
    
    def _parse_contracts_fast(self, root: ET.Element, filing_metadata: Dict[str, Any]) -> pd.DataFrame:
        """Fast contract parsing for small files."""
        return self._parse_contracts(root, filing_metadata)  # Reuse existing method
    
    def _parse_contract_products_fast(self, root: ET.Element, filing_metadata: Dict[str, Any]) -> pd.DataFrame:
        """Fast contract product parsing for small files."""
        return self._parse_contract_products(root, filing_metadata)  # Reuse existing method
    
    def _parse_transactions_chunked(self, root: ET.Element, filing_metadata: Dict[str, Any]) -> pd.DataFrame:
        """Parse transactions with chunking for memory efficiency."""
        return self._parse_transactions(root, filing_metadata)  # Use our improved chunked method
    
    # === STREAMING DATA EXTRACTION METHODS ===
    
    def _extract_organization_data(self, elem: ET.Element, filing_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract organization data from XML element."""
        return {
            'organization_uid': elem.get('Uid'),
            'cid': elem.get('Cid'),
            'company_name': elem.get('Name'),
            'is_filer': elem.get('IsFiler', 'false').lower() == 'true',
            'is_buyer': elem.get('IsBuyer', 'false').lower() == 'true',
            'is_seller': elem.get('IsSeller', 'false').lower() == 'true',
            'transactions_reported_to_index_publisher': elem.get('TransactionsReportedToIndexPublisher', 'false').lower() == 'true',
            'filing_uid': filing_metadata['filing_uid'],
            'year': filing_metadata['year'],
            'quarter': filing_metadata['quarter'],
            'period_type': filing_metadata['period_type'],
            'filing_type': filing_metadata['filing_type']
        }
    
    def _extract_contact_data(self, elem: ET.Element, filing_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract contact data from XML element."""
        # Find parent organization
        org_elem = elem.getparent()
        while org_elem is not None and not (org_elem.tag.endswith('}Organization') or org_elem.tag == 'Organization'):
            org_elem = org_elem.getparent()
        
        organization_uid = org_elem.get('Uid') if org_elem is not None else None
        
        return {
            'contact_uid': elem.get('Uid'),
            'organization_uid': organization_uid,
            'first_name': elem.get('FirstName'),
            'last_name': elem.get('LastName'),
            'display_name': elem.get('DisplayName'),
            'title': elem.get('Title'),
            'phone': elem.get('Phone'),
            'email': elem.get('Email'),
            'is_filer_contact': elem.get('IsFilerContact', 'false').lower() == 'true',
            'is_buyer_contact': elem.get('IsBuyerContact', 'false').lower() == 'true',
            'is_seller_contact': elem.get('IsSellerContact', 'false').lower() == 'true',
            'filing_uid': filing_metadata['filing_uid'],
            'year': filing_metadata['year'],
            'quarter': filing_metadata['quarter'],
            'period_type': filing_metadata['period_type'],
            'filing_type': filing_metadata['filing_type'],
            'street1': elem.get('Street1'),
            'street2': elem.get('Street2'), 
            'street3': elem.get('Street3'),
            'city': elem.get('City'),
            'state': elem.get('State'),
            'zip': elem.get('Zip'),
            'country': elem.get('Country')
        }
    
    def _extract_contract_data(self, elem: ET.Element, filing_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract contract data from XML element."""
        return {
            'contract_uid': elem.get('Uid'),
            'seller_uid': elem.get('SellerUid'),
            'buyer_uid': elem.get('BuyerUid'),
            'ferc_tariff_reference': elem.get('FercTariffReference'),
            'contract_service_agreement': elem.get('ContractServiceAgreement'),
            'is_affiliate': elem.get('IsAffiliate', 'false').lower() == 'true',
            'execution_date': elem.get('ExecutionDate'),
            'commencement_date': elem.get('CommencementDate'),
            'termination_date': elem.get('TerminationDate'),
            'extension_provision_description': elem.get('ExtensionProvisionDescription'),
            'filing_type_contract': elem.get('FilingType'),
            'filing_uid': filing_metadata['filing_uid'],
            'year': filing_metadata['year'],
            'quarter': filing_metadata['quarter'],
            'period_type': filing_metadata['period_type'],
            'filing_type': filing_metadata['filing_type']
        }
    
    def _extract_contract_product_data(self, elem: ET.Element, filing_metadata: Dict[str, Any], contract_uid: str) -> Dict[str, Any]:
        """Extract contract product data from XML element."""
        rate_elem = elem.find('eqr:Rate', self.namespace)
        if rate_elem is None:
            rate_elem = elem.find('Rate')
        return {
            'product_uid': elem.get('Uid'),
            'contract_uid': contract_uid,
            'product_type': elem.get('ProductType'),
            'product_name': elem.get('ProductName'),
            'product_class': elem.get('Class'),
            'term': elem.get('Term'),
            'increment': elem.get('Increment'),
            'increment_peaking': elem.get('IncrementPeaking'),
            'quantity': elem.get('Quantity'),
            'units': elem.get('Units'),
            'podsl': elem.get('Podsl'),
            'begin_date': elem.get('BeginDate'),
            'end_date': elem.get('EndDate'),
            'rate_description': rate_elem.get('Description') if rate_elem is not None else None,
            'rate_units': rate_elem.get('Units') if rate_elem is not None else None,
            'filing_type_product': elem.get('FilingType'),
            'filing_uid': filing_metadata['filing_uid'],
            'year': filing_metadata['year'],
            'quarter': filing_metadata['quarter'],
            'period_type': filing_metadata['period_type'],
            'filing_type': filing_metadata['filing_type']
        }
    
    def _extract_transaction_data(self, elem: ET.Element, filing_metadata: Dict[str, Any], contract_uid: str) -> Dict[str, Any]:
        """Extract transaction data from XML element."""
        return {
            'transaction_uid': elem.get('Uid'),
            'contract_uid': contract_uid,
            'transaction_group_ref': elem.get('TransactionGroupRef'),
            'begin_date': elem.get('BeginDate'),
            'end_date': elem.get('EndDate'),
            'time_zone': elem.get('TimeZone'),
            'trade_date': elem.get('TradeDate'),
            'podba': elem.get('Podba'),
            'podsl': elem.get('Podsl'),
            'transaction_class': elem.get('Class'),
            'term': elem.get('Term'),
            'increment': elem.get('Increment'),
            'increment_peaking': elem.get('IncrementPeaking'),
            'product_name': elem.get('ProductName'),
            'quantity': elem.get('Quantity'),
            'standardized_quantity': elem.get('StandardizedQuantity'),
            'price': elem.get('Price'),
            'standardized_price': elem.get('StandardizedPrice'),
            'rate_units': elem.get('RateUnits'),
            'rate_type': elem.get('RateType'),
            'total_transmission_charge': elem.get('TotalTransmissionCharge'),
            'transaction_charge': elem.get('TransactionCharge'),
            'filing_type_transaction': elem.get('FilingType'),
            'filing_uid': filing_metadata['filing_uid'],
            'year': filing_metadata['year'],
            'quarter': filing_metadata['quarter'],
            'period_type': filing_metadata['period_type'],
            'filing_type': filing_metadata['filing_type']
        }